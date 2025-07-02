export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model Fourier_Transformer \
  --data SMD \
  --data_path SMD_test.npy \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 2 \
  --d_ff 2048 \
  --dropout 0.1\
  --enc_in 38 \
  --c_out 38 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 1