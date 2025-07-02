export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMAP \
  --model_id SMAP \
  --model Fourier_Transformer \
  --data SMAP \
  --data_path SMAP_test.npy \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 2 \
  --d_ff 2048 \
  --dropout 0.1\
  --enc_in 25 \
  --c_out 25 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 1
