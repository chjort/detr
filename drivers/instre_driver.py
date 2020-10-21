import glob
import os

import torch
from torch.utils.data import Dataset, DistributedSampler, DataLoader

from datasets.coco import make_coco_transforms
from datasets.osd_dataset import OSDDataset
from util.misc import collate_fn_os
from util.plot_utils import plot_results

# INSTRE_PATH = "/datadrive/crr/datasets/instre"
INSTRE_PATH = "/home/ch/datasets/instre"
TRAIN_PATH = INSTRE_PATH + "/INSTRE-S-TRAIN"
TEST_PATH = INSTRE_PATH + "/INSTRE-S-TEST"

# %%
BATCH_SIZE = 4
DISTRIBUTED = False
NUM_WORKERS = 2

class_dirs_train = glob.glob(os.path.join(TRAIN_PATH, "*/"))
class_dirs_val = glob.glob(os.path.join(TEST_PATH, "*/"))

dataset_train = OSDDataset(class_dirs_train,
                           make_coco_transforms("train")
                           )
dataset_val = OSDDataset(class_dirs_val,
                         make_coco_transforms("val")
                         )

# %%
if DISTRIBUTED:
    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
else:
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, BATCH_SIZE, drop_last=True)

dataloader_train = DataLoader(dataset_train,
                              batch_sampler=batch_sampler_train,
                              collate_fn=collate_fn_os,
                              num_workers=NUM_WORKERS)

# %%
it = iter(dataloader_train)

# %%
b = next(it)
idx = 0
img = b["queries"][0].tensors[idx]
box = b["queries"][1][idx]["boxes"]
plot_results(img, box)
