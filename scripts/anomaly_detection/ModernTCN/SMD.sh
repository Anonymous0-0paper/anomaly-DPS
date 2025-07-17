export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model ModernTCN \
  --data SMD \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 16 \
  --d_ff 16 \
  --e_layers 2 \
  --enc_in 38 \
  --c_out 38 \
  --top_k 5 \
  --anomaly_ratio 0.5 \
  --batch_size 128 \
  --train_epochs 50