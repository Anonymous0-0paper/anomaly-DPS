export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --gpu 0\
  --root_path ./dataset/msl \
  --model_id MSL \
  --model Transformer \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 501 \
  --c_out 501 \
  --anomaly_ratio 1 \
  --batch_size 64 \
  --train_epochs 10