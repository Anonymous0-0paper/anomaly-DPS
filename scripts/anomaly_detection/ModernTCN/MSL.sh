export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
  --model_id MSL \
  --model ModernTCN \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 16 \
  --d_ff 16 \
  --e_layers 2 \
  --enc_in 55 \
  --c_out 55 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 50