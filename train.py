import time

from triton.interpreter.interpreter import torch

from options.train_options import TrainOptions
from models import create_model
import numpy as np
from image_similarity_measures.quality_metrics import ssim
from skimage.metrics import peak_signal_noise_ratio as psnr  # newer version
import os

from torch.utils.tensorboard import SummaryWriter

torch.multiprocessing.set_sharing_strategy("file_system")

from flairsyn.lib.datasets import create_loaders
from flairsyn.lib.utils.visualization import save_grid
from omegaconf import OmegaConf
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8096, rlimit[1]))


# data_conf = OmegaConf.load("defaults.yml")["data"]


def print_log(logger, message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + "\n")


if __name__ == "__main__":
    opt = TrainOptions().parse()

    data_conf = OmegaConf.load(opt.config)["data"]
    guidance_seqs = data_conf["guidance_sequences"]
    target_seq = data_conf["target_sequence"]

    data_conf["batch_size"] = opt.batchSize

    train_loader, val_loader = create_loaders(**data_conf)

    if len(guidance_seqs) == 1:
        guidance_seqs = [guidance_seqs[0], guidance_seqs[0]]

    ##logger ##
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger = open(os.path.join(save_dir, "log.txt"), "w+")
    print_log(logger, opt.name)
    logger.close()

    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))

    if opt.model == "cycle_gan":
        L1_avg = np.zeros([2, opt.niter + opt.niter_decay, len(val_loader)])
        psnr_avg = np.zeros([2, opt.niter + opt.niter_decay, len(train_loader)])
        ssim_avg = np.zeros([2, opt.niter + opt.niter_decay, len(train_loader)])
    else:
        L1_avg = np.zeros([opt.niter + opt.niter_decay, len(val_loader)])
        psnr_avg = np.zeros([opt.niter + opt.niter_decay, len(val_loader)])
        ssim_avg = np.zeros([opt.niter + opt.niter_decay, len(val_loader)])
    model = create_model(opt)
    # visualizer = Visualizer(opt)
    total_steps = 0
    fixed_val_batch = next(iter(val_loader))

    for epoch in range(opt.epoch_count, (opt.niter + opt.niter_decay + 1)):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        # Training step
        opt.phase = "train"  # ~150 axial slices per volume
        for i, batch in enumerate(tqdm(train_loader)):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            # visualizer.reset()
            total_steps += opt.batchSize  # Note that this already accounts for bs
            epoch_iter += opt.batchSize

            A = torch.cat([batch[seq] for seq in guidance_seqs], dim=1).float()

            data = {
                "A": A.float().contiguous(),
                "B": batch[target_seq].float().contiguous(),
            }

            model.set_input(data)
            model.optimize_parameters()

            errors = model.get_current_errors()
            # log errors as dict to tb
            for k, v in errors.items():
                writer.add_scalar(f"Loss_{k}", v, epoch)

            iter_data_time = time.time()
        # Validaiton step
        if epoch % opt.save_epoch_freq == 0:
            data = {
                "A": torch.concat(
                    [fixed_val_batch[seq] for seq in guidance_seqs], dim=1
                ).float(),
                "B": fixed_val_batch[target_seq].float(),
            }

            model.set_input(data)
            model.test()
            save_grid(model.fake_B, os.path.join(save_dir, f"val_{epoch:04d}.png"), 4)

            logger = open(os.path.join(save_dir, "log.txt"), "a")
            opt.phase = "val"
            for i, val_batch in enumerate(val_loader):
                #
                data = {
                    "A": torch.concat(
                        [val_batch[seq] for seq in guidance_seqs], dim=1
                    ).float(),
                    "B": val_batch[target_seq].float(),
                }

                model.set_input(data)
                #
                model.test()
                #
                fake_im = model.fake_B.cpu().data.numpy()
                real_im = model.real_B.cpu().data.numpy()
                real_im = real_im * 0.5 + 0.5
                fake_im = fake_im * 0.5 + 0.5
                real_im = real_im / real_im.max()
                fake_im = fake_im / fake_im.max()
                if real_im.max() <= 0:
                    continue
                L1_avg[epoch - 1, i] = abs(fake_im - real_im).mean()
                psnr_avg[epoch - 1, i] = psnr(real_im, fake_im, data_range=1)
                ssim_avg[epoch - 1, i] = ssim(
                    real_im[:, 0].transpose(), fake_im[:, 0].transpose(), max_p=1
                )

            l1_avg_loss = np.mean(L1_avg[epoch - 1])
            mean_psnr = np.mean(psnr_avg[epoch - 1])
            std_psnr = np.std(psnr_avg[epoch - 1])

            writer.add_scalar("L1_val", l1_avg_loss, epoch)
            writer.add_scalar("PSNR", mean_psnr, epoch)
            writer.add_scalar("SSIM", np.mean(ssim_avg[epoch - 1]), epoch)

            #
            # print_log(
            #     logger,
            #     "Epoch %3d   l1_avg_loss: %.5f   mean_psnr: %.3f  std_psnr:%.3f "
            #     % (epoch, l1_avg_loss, mean_psnr, std_psnr),
            # )
            # #
            # print_log(logger, "")
            logger.close()
            #
            # print(
            #     "saving the model at the end of epoch %d, iters %d"
            #     % (epoch, total_steps)
            # )
            #
            model.save("latest")
            #
            model.save(epoch)

        # print(
        #     "End of epoch %d / %d \t Time Taken: %d sec"
        #     % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
        # )
        model.update_learning_rate()
