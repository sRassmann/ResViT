PRETRAIN="res_cnn_10k_brats"
FINETUNE="resvit_20k_brats"
INF_BATCH_SIZE=12

DEVICE=1

CUDA_VISIBLE_DEVICES=$DEVICE python train.py --gpu_ids 0 --model resvit_many --which_model_netG res_cnn --lambda_A 100 \
  --dataset_mode aligned --norm batch --pool_size 0 --output_nc 1 --input_nc 3 \
  --checkpoints_dir checkpoints/ --lr 0.0002 --name $PRETRAIN --config brats_train.yml\
  --niter 5000 --niter_decay 5000 --save_epoch_freq 100 --batchSize 24

CUDA_VISIBLE_DEVICES=$DEVICE python train.py --gpu_ids 0 --model resvit_many --which_model_netG resvit --lambda_A 100 \
  --dataset_mode aligned --norm batch --pool_size 0 --output_nc 1 --input_nc 3 \
  --checkpoints_dir checkpoints/ --name $FINETUNE --batchSize 24 --config brats_train.yml\
  --niter 5000 --niter_decay 5000 --save_epoch_freq 100 \
  --pre_trained_transformer 1 --pre_trained_resnet 1 \
  --pre_trained_path checkpoints/$PRETRAIN/latest_net_G.pth --lr 0.0002  # diverged with stated higher LR

CUDA_VISIBLE_DEVICES=$DEVICE python test.py --name $FINETUNE --gpu_ids 0 --model resvit_many --which_model_netG resvit \
  --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 3 --batchSize=$INF_BATCH_SIZE \
  --serial_batches --checkpoints_dir checkpoints/ --which_epoch latest --config brats_train.yml\
  --out_dir_name inference --dataset_json ../data/BraTS/brats23_train.json --data_dir ../data/BraTS/brats23_conformed
#
#CUDA_VISIBLE_DEVICES=$DEVICE python test.py --name $PRETRAIN --gpu_ids 0 --model resvit_many --which_model_netG res_cnn \
#  --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 3 --batchSize=$INF_BATCH_SIZE \
#  --serial_batches --checkpoints_dir checkpoints/ --which_epoch latest --config brats_train.yml\
#  --out_dir_name inference_cnn --dataset_json ../data/BraTS/brats23_train.json --data_dir ../data/BraTS/brats23_conformed