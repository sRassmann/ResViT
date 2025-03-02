PRETRAIN="res_cnn_10k_no_mask"
FINETUNE="resvit_20k_no_mask"
INF_BATCH_SIZE=12

DEVICE=6

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


CUDA_VISIBLE_DEVICES=$DEVICE python test.py --name $PRETRAIN --gpu_ids 0 --model resvit_many --which_model_netG res_cnn \
  --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 3 --batchSize=$INF_BATCH_SIZE \
  --out_dir_name inference_cnn --serial_batches --checkpoints_dir checkpoints/ --which_epoch latest

CUDA_VISIBLE_DEVICES=$DEVICE python test.py --name $FINETUNE --gpu_ids 0 --model resvit_many --which_model_netG resvit \
  --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 3 --batchSize=$INF_BATCH_SIZE \
  --serial_batches --checkpoints_dir checkpoints/ --which_epoch latest

CUDA_VISIBLE_DEVICES=$DEVICE python test.py --name $PRETRAIN --gpu_ids 0 --model resvit_many --which_model_netG res_cnn \
  --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 3 --batchSize=$INF_BATCH_SIZE \
  --serial_batches --checkpoints_dir checkpoints/ --which_epoch latest \
  --out_dir_name inference_wmh_cnn --dataset_json ../data/RS/RS_wmh_test.json --data_dir ../data/RS/conformed_test

CUDA_VISIBLE_DEVICES=$DEVICE python test.py --name $FINETUNE --gpu_ids 0 --model resvit_many --which_model_netG resvit \
  --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 3 --batchSize=$INF_BATCH_SIZE \
  --serial_batches --checkpoints_dir checkpoints/ --which_epoch latest \
  --out_dir_name inference_wmh --dataset_json ../data/RS/RS_wmh_test.json --data_dir ../data/RS/conformed_test

CUDA_VISIBLE_DEVICES=$DEVICE python test.py --name $PRETRAIN --gpu_ids 0 --model resvit_many --which_model_netG res_cnn \
  --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 3 --batchSize=$INF_BATCH_SIZE \
  --serial_batches --checkpoints_dir checkpoints/ --which_epoch latest \
  --out_dir_name inference_pvs_cnn --dataset_json ../data/RS/RS_pvs_test.json --data_dir ../data/RS/conformed_pvs_test

CUDA_VISIBLE_DEVICES=$DEVICE python test.py --name $FINETUNE --gpu_ids 0 --model resvit_many --which_model_netG resvit \
  --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 3 --batchSize=$INF_BATCH_SIZE \
  --serial_batches --checkpoints_dir checkpoints/ --which_epoch latest \
  --out_dir_name inference_pvs --dataset_json ../data/RS/RS_pvs_test.json --data_dir ../data/RS/conformed_pvs_test

CUDA_VISIBLE_DEVICES=$DEVICE python test.py --name $PRETRAIN --gpu_ids 0 --model resvit_many --which_model_netG res_cnn \
  --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 3 --batchSize=$INF_BATCH_SIZE \
  --serial_batches --checkpoints_dir checkpoints/ --which_epoch latest \
  --out_dir_name inference_t_cnn --dataset_json ../data/RS/RS_test.json --data_dir ../data/RS/conformed_test_600

CUDA_VISIBLE_DEVICES=$DEVICE python test.py --name $FINETUNE --gpu_ids 0 --model resvit_many --which_model_netG resvit \
  --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 3 --batchSize=$INF_BATCH_SIZE \
  --serial_batches --checkpoints_dir checkpoints/ --which_epoch latest \
  --out_dir_name inference_t --dataset_json ../data/RS/RS_test.json --data_dir ../data/RS/conformed_test_600

CUDA_VISIBLE_DEVICES=$DEVICE python test.py --name $FINETUNE --gpu_ids 0 --model resvit_many --which_model_netG resvit \
  --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 3 --batchSize=$INF_BATCH_SIZE \
  --serial_batches --checkpoints_dir checkpoints/ --which_epoch latest --config defaults.yml \
  --out_dir_name inference_bmb --dataset_json ../data/test_datasets/bmb_bbreg.json --data_dir ../data/test_datasets/bmb

cd flairsyn
python metrics_3d.py -s checkpoints/$PRETRAIN/inference
python metrics_3d.py -s checkpoints/$FINETUNE/inference_cnn
python metrics_3d.py -s checkpoints/$PRETRAIN/inference_t
python metrics_3d.py -s checkpoints/$FINETUNE/inference_t_cnn
