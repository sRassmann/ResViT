import time

from triton.interpreter.interpreter import torch

from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model

# from util.visualizer import Visualizer
import numpy as np, h5py
from skimage.metrics import peak_signal_noise_ratio as psnr  # newer version
import os

import sys

sys.path.append("..")

from flairsyn.lib.datasets import create_loaders
from omegaconf import OmegaConf

data_conf = OmegaConf.load("../flairsyn/configs/sr3/defaults.yml")["data"]
data_conf["img_size"] = (256, 256)

guidance_seqs = ["t1", "t2"]
target_seq = "flair"


def print_log(logger, message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + "\n")


if __name__ == "__main__":
    opt = TrainOptions().parse()
    data_conf["batch_size"] = opt.batchSize

    train_loader, val_loader = create_loaders(**data_conf)

    ##logger ##
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger = open(os.path.join(save_dir, "log.txt"), "w+")
    print_log(logger, opt.name)
    logger.close()

    if opt.model == "cycle_gan":
        L1_avg = np.zeros([2, opt.niter + opt.niter_decay, len(val_loader)])
        psnr_avg = np.zeros([2, opt.niter + opt.niter_decay, len(train_loader)])
    else:
        L1_avg = np.zeros([opt.niter + opt.niter_decay, len(val_loader)])
        psnr_avg = np.zeros([opt.niter + opt.niter_decay, len(val_loader)])
    model = create_model(opt)
    # visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        # Training step
        opt.phase = "train"
        for _ in range(150):  # ~150 axial slices per volume
            for batch in enumerate(train_loader):
                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                # visualizer.reset()
                total_steps += opt.batchSize  # Note that this already accounts for bs
                epoch_iter += opt.batchSize

                data = {
                    "A": torch.concat(
                        [batch[seq] for seq in guidance_seqs], dim=1
                    ).float(),
                    "B": batch[target_seq].float(),
                }

                model.set_input(data)
                model.optimize_parameters()

                if total_steps % opt.print_freq == 0:
                    errors = model.get_current_errors()

                if total_steps % opt.save_latest_freq == 0:
                    print(
                        "saving the latest model (epoch %d, total_steps %d)"
                        % (epoch, total_steps)
                    )
                    model.save("latest")

            iter_data_time = time.time()
        # Validaiton step
        if epoch % opt.save_epoch_freq == 0:
            logger = open(os.path.join(save_dir, "log.txt"), "a")
            print(opt.dataset_mode)
            opt.phase = "val"
            for i, val_batch in enumerate(val_loader):
                #
                data = {
                    "A": torch.concat(
                        [val_batch[seq] for seq in guidance_seqs], dim=1
                    ).float(),
                    "B": batch[target_seq].float(),
                }

                model.set_input(data)
                #
                model.test()
                #
                fake_im = model.fake_B.cpu().data.numpy()
                #
                real_im = model.real_B.cpu().data.numpy()
                #
                real_im = real_im * 0.5 + 0.5
                #
                fake_im = fake_im * 0.5 + 0.5
                if real_im.max() <= 0:
                    continue
                L1_avg[epoch - 1, i] = abs(fake_im - real_im).mean()
                psnr_avg[epoch - 1, i] = psnr(
                    fake_im / fake_im.max(), real_im / real_im.max()
                )
            #
            #
            l1_avg_loss = np.mean(L1_avg[epoch - 1])
            #
            mean_psnr = np.mean(psnr_avg[epoch - 1])
            #
            std_psnr = np.std(psnr_avg[epoch - 1])
            #
            print_log(
                logger,
                "Epoch %3d   l1_avg_loss: %.5f   mean_psnr: %.3f  std_psnr:%.3f "
                % (epoch, l1_avg_loss, mean_psnr, std_psnr),
            )
            #
            print_log(logger, "")
            logger.close()
            #
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_steps)
            )
            #
            model.save("latest")
            #
            model.save(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
        )
        model.update_learning_rate()
