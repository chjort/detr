import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets.coco import make_coco_transforms_target, make_coco_transforms_query
from datasets.osd_dataset import OSDDataset
from models.backbone import Backbone, Joiner
from models.detr import OSDETR, SetCriterion, PostProcess
from models.matcher import HungarianMatcher
from models.position_encoding import build_position_encoding
from models.transformer import Transformer
from util.plot_utils import plot_results


class Args:
    # backbone
    backbone = "resnet50"
    dilation = False
    position_embedding = "sine"

    # transformer
    num_classes = 91
    enc_layers = 6
    dec_layers = 6
    dim_feedforward = 2048
    hidden_dim = 256
    dropout = 0.1
    nheads = 8
    num_queries = 100
    pre_norm = False

    # training
    batch_size = 1
    aux_loss = True

    output_dir = "outputs"
    device = "cuda"
    seed = 42
    num_workers = 2


# %%
args = Args()
utils.init_distributed_mode(args)

device = torch.device(args.device)

# fix the seed for reproducibility
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# %% PREPARE DATA
INSTRE_PATH = "/datadrive/crr/datasets/instre"
TRAIN_PATH = INSTRE_PATH + "/INSTRE-S-TRAIN"
TEST_PATH = INSTRE_PATH + "/INSTRE-S-TEST"

class_dirs_train = glob.glob(os.path.join(TRAIN_PATH, "*/"))
class_dirs_val = glob.glob(os.path.join(TEST_PATH, "*/"))

n_train_classes = len(class_dirs_train)
n_val_classes = len(class_dirs_val)
n_classes = n_train_classes + n_val_classes

train_labels = list(range(n_train_classes))
val_labels = list(range(n_train_classes, n_train_classes + n_val_classes))

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

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)

batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

data_loader_train = DataLoader(dataset_train,
                               batch_sampler=batch_sampler_train,
                               collate_fn=utils.collate_fn_os,
                               num_workers=args.num_workers)

data_loader_val = DataLoader(dataset_train,
                             args.batch_size,
                             sampler=sampler_val,
                             collate_fn=utils.collate_fn_os,
                             drop_last=False,
                             num_workers=args.num_workers)

# %% BUILD MODEL
position_embedding = build_position_encoding(args)
base_backbone = Backbone(args.backbone, False, False, args.dilation)
backbone = Joiner(base_backbone, position_embedding)
backbone.num_channels = base_backbone.num_channels

transformer = Transformer(
    d_model=args.hidden_dim,
    dropout=args.dropout,
    nhead=args.nheads,
    dim_feedforward=args.dim_feedforward,
    num_encoder_layers=args.enc_layers,
    num_decoder_layers=args.dec_layers,
    normalize_before=args.pre_norm,
    return_intermediate_dec=True,
)
model = OSDETR(
    backbone,
    transformer,
    num_classes=n_classes,
    num_queries=args.num_queries,
    aux_loss=args.aux_loss,
)

model.to(device)
postprocessors = {'bbox': PostProcess()}

# %%
checkpoint = torch.load("outputs/checkpoint.pth", map_location='cpu')
model.load_state_dict(checkpoint['model'])

# %%
it = iter(data_loader_train)

# %%
b = next(it)
qx, qy = b["queries"]
tx, ty = b["targets"]
qx = qx.to(device)
tx = tx.to(device)
ty = [{k: v.to(device) for k, v in t.items()} for t in ty]

outputs = model(qx, tx)

#%%
idx = 0
probas = outputs['pred_logits'].softmax(-1)[idx, :, :-1]
keep = probas.max(-1).values > 0.2

img = tx.tensors[idx].detach().cpu()
boxes = outputs["pred_boxes"][idx, keep].detach().cpu()
boxes = outputs["pred_boxes"][idx, :10].detach().cpu()

plot_results(img, boxes)