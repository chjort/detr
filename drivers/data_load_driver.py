import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets.coco import CocoDetection, make_coco_transforms
from util.plot_utils import plot_results

# %%
BATCH_SIZE = 4
NUM_WORKERS = 2
distributed = False

# COCO_PATH = "/datadrive/crr/datasets/coco"
# COCO_PATH = "/home/ch/datasets/coco"
COCO_PATH = "/home/ch/datasets/synthetic_fruit"

dataset_train = CocoDetection(COCO_PATH + "/train2017",
                              COCO_PATH + "/annotations/instances_train2017.json",
                              transforms=make_coco_transforms("train"),
                              return_masks=False)
dataset_val = CocoDetection(COCO_PATH + "/val2017",
                            COCO_PATH + "/annotations/instances_val2017.json",
                            transforms=make_coco_transforms("val"),
                            return_masks=False)

# %%
if distributed:
    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
else:
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, BATCH_SIZE, drop_last=True)

data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                               collate_fn=utils.collate_fn, num_workers=NUM_WORKERS)
data_loader_val = DataLoader(dataset_val, BATCH_SIZE, sampler=sampler_val,
                             drop_last=False, collate_fn=utils.collate_fn, num_workers=NUM_WORKERS)

# %%
raw_train_it = iter(dataset_train)
raw_val_it = iter(dataset_val)

train_it = iter(data_loader_train)
val_it = iter(data_loader_val)

# %%
x, y = next(raw_val_it)

img = x
box = y["boxes"]

plot_results(img, box)

# %%
x, y = next(val_it)

img = x.tensors[0]
box = y[0]["boxes"]

plot_results(img, box)
