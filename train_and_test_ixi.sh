PRETRAIN="res_cnn_10k_ixiT1"
FINETUNE="resvit_20k_ixiT1"
INF_BATCH_SIZE=4

DEVICE=0

CUDA_VISIBLE_DEVICES=$DEVICE python train.py --gpu_ids 0 --model resvit_many --which_model_netG res_cnn --lambda_A 100 \
  --dataset_mode aligned --norm batch --pool_size 0 --output_nc 1 --input_nc 3 \
  --checkpoints_dir /localmount/volume1/users/rassmanns/ResVit/checkpoints --lr 0.0002 --name $PRETRAIN --config ixi.yml \
  --niter 5000 --niter_decay 5000 --save_epoch_freq 100 --batchSize 12

CUDA_VISIBLE_DEVICES=$DEVICE python train.py --gpu_ids 0 --model resvit_many --which_model_netG resvit --lambda_A 100 \
  --dataset_mode aligned --norm batch --pool_size 0 --output_nc 1 --input_nc 3 \
  --checkpoints_dir /localmount/volume1/users/rassmanns/ResVit/checkpoints --name $FINETUNE --config ixi.yml \
  --niter 5000 --niter_decay 5000 --save_epoch_freq 100 \
  --pre_trained_transformer 1 --pre_trained_resnet 1  --batchSize 24 \
  --pre_trained_path /localmount/volume1/users/rassmanns/ResVit/checkpoints/res_cnn_10k_ixiT1/$PRETRAIN/latest_net_G.pth --lr 0.0002  # diverged with stated higher LR

CUDA_VISIBLE_DEVICES=$DEVICE python test.py --name $FINETUNE --gpu_ids 0 --model resvit_many --which_model_netG resvit \
  --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 3 --batchSize=$INF_BATCH_SIZE \
  --serial_batches --checkpoints_dir /localmount/volume1/users/rassmanns/ResVit/checkpoints/ --which_epoch latest --config ixi.yml \
  --out_dir_name inference --dataset_json ../data/test_datasets/ixi_train.json --data_dir ../data/test_datasets/ixi

CUDA_VISIBLE_DEVICES=$DEVICE python test.py --name $PRETRAIN --gpu_ids 0 --model resvit_many --which_model_netG res_cnn \
  --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 3 --batchSize=$INF_BATCH_SIZE \
  --serial_batches --checkpoints_dir /localmount/volume1/users/rassmanns/ResVit/checkpoints/res_cnn_10k_ixiT1 --which_epoch latest --config ixi.yml \
  --out_dir_name inference_cnn --dataset_json ../data/test_datasets/ixi_train.json --data_dir ../data/test_datasets/ixi
