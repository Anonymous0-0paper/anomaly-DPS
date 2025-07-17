import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import csv
import seaborn as sns
import sys

# 获取当前脚本所在目录作为基础路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "analysis_results")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配置数据集和模型
DATASETS = ["MSL", "SMAP", "SMD", "SWAT"]
MODELS = {
    "gc8": "Autoformer",
    "gc9": "DACAD",
    "gc10": "Fourier_Transformer",
    "gc11": "OmniAnomaly",
    "gc12": "TimesNet",
    "gc13": "TranAD",
    "gc14": "Transformer"
}

# 在输出目录下创建子目录
os.makedirs(os.path.join(OUTPUT_DIR, "metrics"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "loss_curves"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "gpu_usage"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "cross_model"), exist_ok=True)

# 用于存储所有解析结果的数据结构
all_results = []


def print_header(title):
    """打印带格式的标题"""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def parse_training_log(log_path):
    """解析训练日志文件"""
    results = {
        "epochs": [], "train_loss": [], "val_loss": [], "test_loss": [],
        "final_metrics": {}, "train_time": None
    }

    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

            # 解析训练过程
            epoch_pattern = r"Epoch: (\d+), Steps: \d+ \| Train Loss: ([\d\.]+) Vali Loss: ([\d\.]+) Test Loss: ([\d\.]+)"
            epochs = re.findall(epoch_pattern, content)
            for ep in epochs:
                results["epochs"].append(int(ep[0]))
                results["train_loss"].append(float(ep[1]))
                results["val_loss"].append(float(ep[2]))
                results["test_loss"].append(float(ep[3]))

            # 解析最终指标
            metrics_pattern = r"Accuracy : ([\d\.]+), Precision : ([\d\.]+), Recall : ([\d\.]+), F-score : ([\d\.]+)"
            metrics = re.search(metrics_pattern, content)
            if metrics:
                results["final_metrics"] = {
                    "Accuracy": float(metrics.group(1)),
                    "Precision": float(metrics.group(2)),
                    "Recall": float(metrics.group(3)),
                    "F-score": float(metrics.group(4))
                }

            # 解析训练时间
            time_pattern = r"总耗时: (\d+) 秒"
            time_match = re.search(time_pattern, content)
            if time_match:
                results["train_time"] = int(time_match.group(1))
            else:
                # 尝试另一种时间格式
                time_pattern2 = r"real\s+(\d+)m([\d\.]+)s"
                time_match2 = re.search(time_pattern2, content)
                if time_match2:
                    minutes = int(time_match2.group(1))
                    seconds = float(time_match2.group(2))
                    results["train_time"] = minutes * 60 + seconds

    except Exception as e:
        print(f"Error parsing {log_path}: {str(e)}")
        import traceback
        traceback.print_exc()

    return results


def parse_gpu_log(gpu_log_path):
    """解析GPU内存日志文件"""
    gpu_data = []
    try:
        with open(gpu_log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # 找到表头行
        header_line = None
        for i, line in enumerate(lines):
            if "timestamp, name, index" in line:
                header_line = i
                break

        if header_line is None:
            print(f"⚠️ No GPU header found in {os.path.basename(gpu_log_path)}")
            return gpu_data

        # 创建CSV读取器
        csv_content = lines[header_line:]
        reader = csv.DictReader(csv_content)

        print(f"  Found header: {reader.fieldnames}")  # 调试信息

        for row in reader:
            try:
                # 检查索引字段是否存在
                if ' index' not in row:
                    continue

                # 只处理第一个GPU
                if row[' index'].strip() == '0':
                    # 处理时间戳
                    timestamp_str = row['timestamp'].strip()

                    # 尝试两种时间格式
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y/%m/%d %H:%M:%S.%f')
                    except ValueError:
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                        except ValueError:
                            print(f"⚠️ Unsupported timestamp format: {timestamp_str}")
                            continue

                    # 处理内存使用
                    memory_used = float(row[' memory.used [MiB]'].split()[0])  # 提取数值部分

                    # 处理GPU利用率
                    gpu_util = float(row[' utilization.gpu [%]'].split()[0])  # 提取数值部分

                    gpu_data.append({
                        "timestamp": timestamp,
                        "memory_used": memory_used,
                        "gpu_util": gpu_util
                    })
            except (ValueError, KeyError) as e:
                print(f"Error processing GPU row: {str(e)}")
                continue

        # 计算时间差（秒）
        if gpu_data:
            start_time = gpu_data[0]["timestamp"]
            for entry in gpu_data:
                delta = entry["timestamp"] - start_time
                entry["time_sec"] = delta.total_seconds()

            print(f"  Processed {len(gpu_data)} GPU records")
        else:
            print("No GPU data found after parsing")

    except Exception as e:
        print(f"Error parsing GPU log {os.path.basename(gpu_log_path)}: {str(e)}")
        import traceback
        traceback.print_exc()

    return gpu_data



def analyze_dataset(dataset):
    """分析单个数据集的所有模型"""
    dataset_dir = os.path.join(BASE_DIR, dataset)
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return

    print_header(f"Analyzing dataset: {dataset}")

    # 存储所有模型的指标
    metrics_data = []
    loss_data = []
    gpu_usage_data = []

    for prefix, model_name in MODELS.items():
        log_file = f"{prefix}_{model_name}.sh.log"
        gpu_log_file = f"{prefix}_{model_name}.sh_gpu_mem.log"
        log_path = os.path.join(dataset_dir, log_file)
        gpu_log_path = os.path.join(dataset_dir, "gpu_mem", gpu_log_file)

        # 检查文件是否存在
        if not os.path.exists(log_path):
            print(f"[WARNING] Log file not found: {log_path}")
            continue
        if not os.path.exists(gpu_log_path):
            print(f"[WARNING] GPU log file not found: {gpu_log_path}")
            continue

        print(f"\nProcessing model: {model_name}")

        # 解析日志
        train_results = parse_training_log(log_path)
        gpu_results = parse_gpu_log(gpu_log_path)

        if not train_results["epochs"]:
            print(f"  [WARNING] No training data found for {model_name}")
            continue
        else:
            print(f"  Found {len(train_results['epochs'])} epochs of training data")

        # 保存结果到全局数据结构
        all_results.append({
            "dataset": dataset,
            "model": model_name,
            **train_results
        })

        # 添加到指标列表
        if train_results["final_metrics"]:
            metrics_entry = {
                "Model": model_name,
                **train_results["final_metrics"]
            }
            if train_results["train_time"] is not None:
                metrics_entry["Train Time (s)"] = train_results["train_time"]
                print(f"  Training time: {train_results['train_time']} seconds")
            metrics_data.append(metrics_entry)

        # 添加损失曲线数据
        for epoch, train_loss, val_loss, test_loss in zip(
                train_results["epochs"],
                train_results["train_loss"],
                train_results["val_loss"],
                train_results["test_loss"]
        ):
            loss_data.append({
                "Dataset": dataset,
                "Model": model_name,
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Validation Loss": val_loss,
                "Test Loss": test_loss
            })

        # 添加GPU使用数据
        if gpu_results:
            print(f"  Found {len(gpu_results)} GPU data points")

            # 计算GPU使用指标
            memory_values = [d["memory_used"] for d in gpu_results]
            util_values = [d["gpu_util"] for d in gpu_results]

            if memory_values and util_values:
                gpu_entry = {
                    "Model": model_name,
                    "Peak Memory (MiB)": max(memory_values),
                    "Avg Memory (MiB)": np.mean(memory_values),
                    "Max GPU Util (%)": max(util_values),
                    "Avg GPU Util (%)": np.mean(util_values)
                }
                gpu_usage_data.append(gpu_entry)
                print(
                    f"  GPU Usage: Peak {gpu_entry['Peak Memory (MiB)']:.1f} MiB, Avg {gpu_entry['Avg GPU Util (%)']:.1f}%")

            # 保存原始GPU数据
            gpu_df = pd.DataFrame(gpu_results)
            gpu_csv_path = os.path.join(OUTPUT_DIR, "gpu_usage", f"{dataset}_{model_name}_gpu.csv")
            gpu_df.to_csv(gpu_csv_path, index=False)
            print(f"  Saved GPU data to: {gpu_csv_path}")

            # 绘制GPU使用曲线
            plt.figure(figsize=(12, 6))
            plt.plot(gpu_df["time_sec"], gpu_df["memory_used"], label="Memory Used (MiB)")
            plt.plot(gpu_df["time_sec"], gpu_df["gpu_util"], label="GPU Utilization (%)")
            plt.title(f"{model_name} GPU Usage - {dataset}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Usage")
            plt.legend()
            plt.grid(True)

            gpu_plot_path = os.path.join(OUTPUT_DIR, "gpu_usage", f"{dataset}_{model_name}_gpu_usage.png")
            plt.savefig(gpu_plot_path)
            plt.close()
            print(f"  Saved GPU plot to: {gpu_plot_path}")
        else:
            print("  No GPU data found")

    # 1. 保存并绘制性能指标对比
    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        metrics_csv_path = os.path.join(OUTPUT_DIR, "metrics", f"{dataset}_metrics.csv")
        df_metrics.to_csv(metrics_csv_path, index=False)
        print(f"\nSaved metrics data to: {metrics_csv_path}")

        # 绘制指标对比图
        if not df_metrics.empty:
            metrics_to_plot = ["Accuracy", "Precision", "Recall", "F-score"]

            # 创建指标对比图
            plt.figure(figsize=(12, 8))
            df_metrics.set_index("Model")[metrics_to_plot].plot(
                kind="bar", rot=45
            )
            plt.title(f"{dataset} - Model Performance Comparison")
            plt.ylabel("Score")
            plt.tight_layout()

            metrics_plot_path = os.path.join(OUTPUT_DIR, "metrics", f"{dataset}_performance_comparison.png")
            plt.savefig(metrics_plot_path)
            plt.close()
            print(f"Saved performance comparison plot to: {metrics_plot_path}")

            # 创建训练时间图
            if "Train Time (s)" in df_metrics.columns:
                plt.figure(figsize=(10, 6))
                df_metrics.set_index("Model")["Train Time (s)"].plot(kind="bar", rot=45)
                plt.title(f"{dataset} - Training Time Comparison")
                plt.ylabel("Time (seconds)")
                plt.tight_layout()

                time_plot_path = os.path.join(OUTPUT_DIR, "metrics", f"{dataset}_training_time.png")
                plt.savefig(time_plot_path)
                plt.close()
                print(f"Saved training time plot to: {time_plot_path}")

    # 2. 保存并绘制损失曲线对比
    if loss_data:
        df_loss = pd.DataFrame(loss_data)
        loss_csv_path = os.path.join(OUTPUT_DIR, "loss_curves", f"{dataset}_loss_data.csv")
        df_loss.to_csv(loss_csv_path, index=False)
        print(f"Saved loss data to: {loss_csv_path}")

        # 绘制损失曲线
        if not df_loss.empty:
            plt.figure(figsize=(12, 8))

            # 为每个模型绘制损失曲线
            for model in df_loss["Model"].unique():
                model_data = df_loss[df_loss["Model"] == model]
                plt.plot(model_data["Epoch"], model_data["Train Loss"], label=f"{model} Train")
                plt.plot(model_data["Epoch"], model_data["Validation Loss"], "--", label=f"{model} Val")

            plt.title(f"{dataset} - Training Progress")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(ncol=2, loc='upper right')
            plt.grid(True)
            plt.tight_layout()

            loss_plot_path = os.path.join(OUTPUT_DIR, "loss_curves", f"{dataset}_loss_curves.png")
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"Saved loss curves plot to: {loss_plot_path}")

    # 3. 保存并绘制GPU资源使用
    if gpu_usage_data:
        df_gpu = pd.DataFrame(gpu_usage_data)
        gpu_csv_path = os.path.join(OUTPUT_DIR, "gpu_usage", f"{dataset}_gpu_summary.csv")
        df_gpu.to_csv(gpu_csv_path, index=False)
        print(f"Saved GPU summary to: {gpu_csv_path}")

        # 绘制GPU资源使用对比
        if not df_gpu.empty:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"{dataset} - GPU Resource Usage", fontsize=16)

            # 峰值内存使用
            df_gpu.plot.bar(x="Model", y="Peak Memory (MiB)", ax=axes[0, 0], rot=45)
            axes[0, 0].set_title("Peak GPU Memory Usage")
            axes[0, 0].set_ylabel("Memory (MiB)")

            # 平均内存使用
            df_gpu.plot.bar(x="Model", y="Avg Memory (MiB)", ax=axes[0, 1], rot=45)
            axes[0, 1].set_title("Average GPU Memory Usage")
            axes[0, 1].set_ylabel("Memory (MiB)")

            # 最大GPU利用率
            df_gpu.plot.bar(x="Model", y="Max GPU Util (%)", ax=axes[1, 0], rot=45)
            axes[1, 0].set_title("Maximum GPU Utilization")
            axes[1, 0].set_ylabel("Utilization (%)")

            # 平均GPU利用率
            df_gpu.plot.bar(x="Model", y="Avg GPU Util (%)", ax=axes[1, 1], rot=45)
            axes[1, 1].set_title("Average GPU Utilization")
            axes[1, 1].set_ylabel("Utilization (%)")

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            gpu_plot_path = os.path.join(OUTPUT_DIR, "gpu_usage", f"{dataset}_gpu_comparison.png")
            plt.savefig(gpu_plot_path)
            plt.close()
            print(f"Saved GPU comparison plot to: {gpu_plot_path}")


def perform_cross_model_analysis():
    """执行跨模型比较分析"""
    print_header("Performing cross-model analysis")

    # 从所有结果中提取指标
    metrics_list = []
    for res in all_results:
        if res.get("final_metrics") and res.get("dataset") and res.get("model"):
            metrics_entry = {
                "Dataset": res["dataset"],
                "Model": res["model"],
                **res["final_metrics"]
            }
            if res.get("train_time"):
                metrics_entry["Train Time (s)"] = res["train_time"]

            metrics_list.append(metrics_entry)

    if not metrics_list:
        print("No metrics data found for cross-model analysis")
        return

    df_metrics = pd.DataFrame(metrics_list)

    # 保存所有指标数据
    all_metrics_path = os.path.join(OUTPUT_DIR, "all_metrics.csv")
    df_metrics.to_csv(all_metrics_path, index=False)
    print(f"Saved all metrics data to: {all_metrics_path}")

    # 1. 跨数据集模型性能对比
    metrics = ["Accuracy", "Precision", "Recall", "F-score"]

    for metric in metrics:
        plt.figure(figsize=(14, 8))
        sns.barplot(x="Model", y=metric, hue="Dataset", data=df_metrics, palette="viridis")
        plt.title(f"Cross-Dataset {metric} Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = os.path.join(OUTPUT_DIR, "cross_model", f"cross_dataset_{metric.lower()}_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {metric} comparison plot to: {plot_path}")

    # 2. 跨模型平均性能
    avg_metrics = df_metrics.groupby("Model")[metrics].mean().reset_index()
    avg_metrics_path = os.path.join(OUTPUT_DIR, "cross_model", "average_metrics.csv")
    avg_metrics.to_csv(avg_metrics_path, index=False)
    print(f"Saved average metrics to: {avg_metrics_path}")

    # 绘制平均性能柱状图
    plt.figure(figsize=(12, 8))
    avg_metrics.set_index("Model").plot(kind="bar", rot=45)
    plt.title("Average Performance Across Datasets")
    plt.ylabel("Score")
    plt.tight_layout()

    avg_plot_path = os.path.join(OUTPUT_DIR, "cross_model", "average_performance.png")
    plt.savefig(avg_plot_path)
    plt.close()
    print(f"Saved average performance plot to: {avg_plot_path}")

    # 3. 训练时间对比
    if "Train Time (s)" in df_metrics.columns:
        # 训练时间对比
        plt.figure(figsize=(12, 8))
        sns.boxplot(x="Model", y="Train Time (s)", data=df_metrics, palette="Set2")
        plt.title("Training Time Comparison Across Datasets")
        plt.xticks(rotation=45)
        plt.tight_layout()

        time_path = os.path.join(OUTPUT_DIR, "cross_model", "training_time_comparison.png")
        plt.savefig(time_path)
        plt.close()
        print(f"Saved training time comparison to: {time_path}")

    # 4. 综合性能-时间散点图
    if "F-score" in df_metrics.columns and "Train Time (s)" in df_metrics.columns:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x="Train Time (s)", y="F-score", hue="Model",
                        size="Accuracy", sizes=(50, 200), data=df_metrics, alpha=0.8)
        plt.title("Performance vs Training Time")
        plt.grid(True)
        plt.tight_layout()

        scatter_path = os.path.join(OUTPUT_DIR, "cross_model", "performance_vs_time.png")
        plt.savefig(scatter_path)
        plt.close()
        print(f"Saved performance vs time plot to: {scatter_path}")


def main():
    # 检查日志目录是否存在
    if not os.path.exists(BASE_DIR):
        print(f"Error: Logs directory not found at {BASE_DIR}")
        print("Please ensure the 'logs' directory exists in the same folder as this script.")
        sys.exit(1)

    print(f"Starting analysis...")
    print(f"Logs directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # 处理所有数据集
    for dataset in DATASETS:
        analyze_dataset(dataset)

    # 执行跨模型分析
    perform_cross_model_analysis()

    print_header("Analysis complete!")
    print(f"All results have been saved to: {OUTPUT_DIR}")
    print("\nSummary of generated files:")
    print("  - Dataset-specific analysis in subdirectories")
    print("  - Cross-model comparison in 'cross_model' directory")
    print("  - Comprehensive metrics in 'all_metrics.csv'")


if __name__ == "__main__":
    # 设置美观的绘图风格
    sns.set_theme(style="whitegrid", palette="pastel")
    plt.rcParams['font.family'] = 'DejaVu Sans'

    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)