export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/wadi \
  --model_id WADI \
  --model Transformer \
  --data WADI \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 8 \
  --d_ff 8 \
  --e_layers 3 \
  --enc_in 119 \
  --c_out 119 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 512 \
  --train_epochs 10