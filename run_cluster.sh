#!/bin/bash

# 0. Set virtual environment path
VENV_PATH="$HOME/project/venvs/anomaly-DPS"
echo "Using virtual environment: $VENV_PATH"

# 1. Set log directory
LOG_BASE="logs/MSL"
echo "Setting log directory: $LOG_BASE"
mkdir -p "$LOG_BASE" || { echo "Unable to create log directory: $LOG_BASE"; exit 1; }
mkdir -p "$LOG_BASE/gpu_mem" || { echo "Unable to create GPU log directory"; exit 1; }

# 2. Ensure all nodes have the same environment
echo "Syncing environment to all nodes..."
for node in gc{8..16}; do
    echo ">>> Configuring environment on $node"

    # Create virtual environment (if it doesn't exist)
    ssh -o "StrictHostKeyChecking=accept-new" "$node" "
        # Create virtual environment directory
        mkdir -p '$VENV_PATH' || true

        # Check if virtual environment is already created
        if [ ! -f '$VENV_PATH/bin/activate' ]; then
            echo 'Creating virtual environment...'
            python3 -m venv '$VENV_PATH'
        fi

        # Activate environment and install dependencies
        source '$VENV_PATH/bin/activate'
        pip install --upgrade pip
        pip install -r '$PWD/requirements.txt'
    " > "$LOG_BASE/${node}_env_setup.log" 2>&1

    echo "  Environment configuration completed, log: $LOG_BASE/${node}_env_setup.log"
done

# 3. Generate node list
echo "Generating node list..."
echo gc{8..16} | tr ' ' '\n' > node_list.txt
nodes_count=$(wc -l < node_list.txt)
echo "Number of available nodes: $nodes_count"

# 4. Prepare script list
echo "Preparing task scripts..."
script_dir="scripts/anomaly_detection/MSL"
if [ ! -d "$script_dir" ]; then
    echo "Error: Script directory $script_dir does not exist!"
    exit 1
fi
# Get script file list
script_files=($(ls "$script_dir"/*.sh 2>/dev/null))
scripts_count=${#script_files[@]}
if [ $scripts_count -eq 0 ]; then
    echo "Error: No scripts found in $script_dir directory!"
    exit 1
fi
echo "Found $scripts_count scripts:"
for script in "${script_files[@]}"; do
    echo "  - $(basename "$script")"
done

# 5. Create task assignments
echo "Assigning tasks to nodes..."
rm -f task_assignments.txt
touch task_assignments.txt
for ((i=0; i<scripts_count; i++)); do
    node=$(head -n $((i+1)) node_list.txt | tail -n 1)
    script_path=${script_files[$i]}
    script_name=$(basename "$script_path")
    echo "$node $script_path" >> task_assignments.txt
    echo "  Node $node ← Script $script_name"
done

# 6. Execute all tasks in parallel (with GPU monitoring)
echo "Starting cluster tasks (with GPU memory monitoring)..."
start_time=$(date +%s)
while read -r node script_path; do
    script_name=$(basename "$script_path")
    log_file="$LOG_BASE/${node}_${script_name}.log"
    gpu_log="$LOG_BASE/gpu_mem/${node}_${script_name}_gpu_mem.log"

    echo "[$(date '+%T')] Starting on $node: $script_name"
    echo "  Task log: $log_file"
    echo "  GPU log: $gpu_log"

    # Execute task on the node (with environment activation)
    ssh -o "StrictHostKeyChecking=accept-new" "$node" "
        # Change to project directory
        cd '$PWD'

        # Activate virtual environment
        source '$VENV_PATH/bin/activate'

        # Check if Python is available
        echo 'Python path: ' \$(which python)
        echo 'Python version: ' \$(python --version)

        # Start GPU memory monitoring
        echo '=== GPU Monitoring Started === ' > '$gpu_log'
        nohup nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit,fan.speed,clocks.current.graphics,clocks.current.memory,pstate --format=csv -l 5 >> '$gpu_log' 2>&1 &
        MONITOR_PID=\$!

        # Execute main task script
        echo '=== Task Started === ' | tee -a '$log_file'
        echo 'Script path: $script_path' | tee -a '$log_file'
        time bash '$script_path' 2>&1 | tee -a '$log_file'

        # Save task exit status
        TASK_EXIT_STATUS=\$?

        # Stop monitoring
        kill \$MONITOR_PID

        echo '=== Task Ended === ' | tee -a '$log_file'
        echo 'Exit status: '\$TASK_EXIT_STATUS | tee -a '$log_file'
        echo 'Total time: '\$SECONDS' seconds' | tee -a '$log_file'
        exit \$TASK_EXIT_STATUS
    " > "$log_file" 2>&1 &

    # Record process ID
    echo $! > "$LOG_BASE/${node}_${script_name}.pid"
done < task_assignments.txt

# 7. Wait for tasks to complete
echo "Waiting for tasks to complete..."
echo "================================================="
echo "  Real-time monitoring commands:"
echo "  View task logs:    tail -f $LOG_BASE/<node_name>_<script_name>.log"
echo "  View GPU memory logs: tail -f $LOG_BASE/gpu_mem/<node_name>_<script_name>_gpu_mem.log"
echo "  Check Python environment:  grep 'Python path' $LOG_BASE/*.log"
echo "================================================="
wait
end_time=$(date +%s)

# 8. Generate report
echo "======================================"
echo "All node tasks completed!"
echo "Total time: $((end_time - start_time)) seconds"
echo ""
echo "Task logs:"
ls -lh "$LOG_BASE"/*.log 2>/dev/null
echo ""
echo "GPU memory monitoring logs:"
ls -lh "$LOG_BASE"/gpu_mem/* 2>/dev/null
echo "======================================"

# 9. Check task status and Python environment
echo ""
echo "Python environment check:"
grep "Python path" "$LOG_BASE"/*.log
echo ""
echo "Task status check:"
for log in "$LOG_BASE"/*.log; do
    exit_status=$(grep 'Exit status:' "$log" | awk '{print $NF}' | tail -1)
    script_name=$(basename "$log")
    echo "  $script_name: Exit status ${exit_status:-unknown}"
done
echo ""
echo "All logs are saved in: $LOG_BASE"
