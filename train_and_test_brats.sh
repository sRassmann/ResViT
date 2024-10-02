PRETRAIN="res_cnn_10k_brats"
FINETUNE="resvit_20k_brats"
INF_BATCH_SIZE=12

DEVICE=1

CUDA_VISIBLE_DEVICES=$DEVICE python train.py --gpu_ids 0 --model resvit_many --which_model_netG res_cnn --lambda_A 100 \
  --dataset_mode aligned --norm batch --pool_size 0 --output_nc 1 --input_nc 3 \
  --checkpoints_dir checkpoints/ --lr 0.0002 --name $PRETRAIN \
  --niter 5000 --niter_decay 5000 --save_epoch_freq 100 --batchSize 24

CUDA_VISIBLE_DEVICES=$DEVICE python train.py --gpu_ids 0 --model resvit_many --which_model_netG resvit --lambda_A 100 \
  --dataset_mode aligned --norm batch --pool_size 0 --output_nc 1 --input_nc 3 \
  --checkpoints_dir checkpoints/ --name $FINETUNE --batchSize 24 \
  --niter 5000 --niter_decay 5000 --save_epoch_freq 100 \
  --pre_trained_transformer 1 --pre_trained_resnet 1 \
  --pre_trained_path checkpoints/$PRETRAIN/latest_net_G.pth --lr 0.0002  # diverged with stated higher LR
