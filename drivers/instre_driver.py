import glob
import os

import torch
from torch.utils.data import Dataset, DistributedSampler, DataLoader

from datasets.coco import make_coco_transforms_query, make_coco_transforms_target
from datasets.osd_dataset import OSDDataset
from util.misc import collate_fn_os
from util.plot_utils import plot_results
import matplotlib.pyplot as plt

INSTRE_PATH = "/datadrive/crr/datasets/instre"

# INSTRE_PATH = "/home/ch/datasets/instre"
TRAIN_PATH = INSTRE_PATH + "/INSTRE-S-TRAIN"
TEST_PATH = INSTRE_PATH + "/INSTRE-S-TEST"

# %%
BATCH_SIZE = 4
DISTRIBUTED = False
NUM_WORKERS = 2

class_dirs_train = glob.glob(os.path.join(TRAIN_PATH, "*/"))
class_dirs_val = glob.glob(os.path.join(TEST_PATH, "*/"))

train_labels = list(range(len(class_dirs_train)))
val_labels = list(range(len(class_dirs_val)))

dataset_train = OSDDataset(class_dirs_train,
                           labels=train_labels,
                           query_transforms=make_coco_transforms_query("train"),
                           target_transforms=make_coco_transforms_target("train")
                           )
dataset_val = OSDDataset(class_dirs_val,
                         labels=val_labels,
                         query_transforms=make_coco_transforms_query("val"),
                         target_transforms=make_coco_transforms_target("val")
                         )

# %%
if DISTRIBUTED:
    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
else:
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, BATCH_SIZE, drop_last=True)

data_loader_train = DataLoader(dataset_train,
                               batch_sampler=batch_sampler_train,
                               collate_fn=collate_fn_os,
                               num_workers=NUM_WORKERS)

data_loader_val = DataLoader(dataset_train,
                             BATCH_SIZE,
                             sampler=sampler_val,
                             collate_fn=collate_fn_os,
                             drop_last=False,
                             num_workers=NUM_WORKERS)

# %%
# it = iter(data_loader_train)
it = iter(data_loader_val)

# %%
b = next(it)

# %%
idx = 1
qimg = b["queries"][0].tensors[idx]
timg = b["targets"][0].tensors[idx]
tbox = b["targets"][1][idx]["boxes"]

plt.imshow(qimg.permute(2, 1, 0))
plt.show()
plot_results(timg, tbox)
