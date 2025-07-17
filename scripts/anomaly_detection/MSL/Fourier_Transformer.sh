export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/msl \
  --model_id MSL \
  --model Fourier_Transformer \
  --data_path MSL_test.npy \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --n_heads 4 \
  --e_layers 2 \
  --d_ff 128 \
  --dropout 0.1\
  --enc_in 55 \
  --c_out 55 \
  --anomaly_ratio 1 \
  --batch_size 64 \
  --train_epochs 1