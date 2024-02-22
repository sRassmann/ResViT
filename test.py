import os

import torch

from options.test_options import TestOptions
from models import create_model

from flairsyn.lib.datasets import get_datasets
from flairsyn.lib.inference import save_output_volume
from omegaconf import OmegaConf
from tqdm import tqdm

config = OmegaConf.load("defaults.yml")
guidance_seqs = ["t1", "t2"]
target_seq = "flair"

if __name__ == "__main__":
    opt = TestOptions().parse()
    # opt.nThreads = 1   # test code only supports nThreads = 1
    # opt.batchSize = 1  # test code only supports batchSize = 1
    # opt.serial_batches = True  # no shuffle
    # opt.no_flip = True  # no flip

    output_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.out_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving predictions to: {output_dir}")

    _, val = get_datasets(
        dataset=opt.dataset_json,
        data_dir=opt.data_dir,
        relevant_sequences=config.data.guidance_sequences
        + [config.data.target_sequence],
        size=None,
        cache=None,
        subset_train=0,
        normalize_to=(-1, 1),
        skull_strip=1,
    )

    model = create_model(opt)

    # test
    for i, vol in enumerate(tqdm(val)):
        pred_vol = torch.zeros_like(vol[target_seq])
        guidance = torch.cat([vol[seq] for seq in guidance_seqs], dim=0).float()
        guidance = guidance.permute(1, 0, 2, 3)

        for j in range(0, vol[target_seq].shape[1], opt.batchSize):
            g = guidance[j : j + opt.batchSize]
            b, c, h, w = g.shape

            # pad to batch size, 2, 256, 256
            pad = -torch.ones([b, c, 256, 256])
            offset_h = (256 - h) // 2
            offset_w = (256 - w) // 2
            pad[:, :, offset_h : offset_h + h, offset_w : offset_w + w] = g

            data = {
                "A": pad,
                "B": pad[:, :1, :, :],  # dummy
            }
            model.set_input(data)
            model.test()
            pred_vol[0, j : j + opt.batchSize] = model.fake_B[
                :, 0, offset_h : offset_h + h, offset_w : offset_w + w
            ].cpu()

        vol["pred"] = torch.clamp(pred_vol, -1, 1)
        save_output_volume(
            vol,
            output_path=output_dir,
            save_keys=list(config.data.guidance_sequences)
            + ["pred", config.data.target_sequence, "mask"],
            target_sequence=config.data.target_sequence,
        )
