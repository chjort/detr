import datetime
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets.coco import make_coco_transforms_target, make_coco_transforms_query
from datasets.osd_dataset import OSDDataset
from engine import train_one_epoch_os
from models.backbone import Backbone, Joiner
from models.detr import OSDETR, SetCriterion, PostProcess
from models.matcher import HungarianMatcher
from models.position_encoding import build_position_encoding
from models.transformer import Transformer
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


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
    batch_size = 2
    epochs = 150  # 300
    lr = 1e-4
    lr_backbone = 1e-5
    weight_decay = 1e-4
    lr_drop = 100  # 200
    clip_max_norm = 0.1

    # resume = "detr-r50-e632da11.pth"
    resume = ""

    # loss
    aux_loss = True
    set_cost_class = 1
    set_cost_bbox = 5
    set_cost_giou = 2
    dice_loss_coef = 1
    bbox_loss_coef = 5
    giou_loss_coef = 2
    eos_coef = 0.1

    # data
    #   (coco)
    coco_path = "/datadrive/crr/datasets/coco"

    output_dir = "outputs"
    device = "cuda"
    seed = 42
    start_epoch = 0
    eval = False
    num_workers = 2

    # distributed
    distributed = False
    world_size = 1
    dist_url = "env://"


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

if args.distributed:
    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
else:
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
train_backbone = args.lr_backbone > 0
base_backbone = Backbone(args.backbone, train_backbone, False, args.dilation)
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
matcher = HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)

weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
weight_dict['loss_giou'] = args.giou_loss_coef
if args.aux_loss:
    aux_weight_dict = {}
    for i in range(args.dec_layers - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

losses = ['labels', 'boxes', 'cardinality']
criterion = SetCriterion(n_classes, matcher=matcher, weight_dict=weight_dict,
                         eos_coef=args.eos_coef, losses=losses)
postprocessors = {'bbox': PostProcess()}

criterion.to(device)
model.to(device)

# %% set distributed model
model_without_ddp = model
if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

# %% set optimizer
param_dicts = [
    {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": args.lr_backbone,
    },
]
optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                              weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

#%% PLAYGROUND
# it = iter(data_loader_train)
# b = next(it)
# qx, qy = b["queries"]
# tx, ty = b["targets"]
# qx = qx.to(device)
# tx = tx.to(device)
# ty = [{k: v.to(device) for k, v in t.items()} for t in ty]
#
# outputs = model(qx, tx)
# targets = ty
#
# bs, num_queries = outputs["pred_logits"].shape[:2]
#
# # We flatten to compute the cost matrices in a batch
# out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
# out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
#
# # Also concat the target labels and boxes
# tgt_ids = torch.cat([v["labels"] for v in targets])
# tgt_bbox = torch.cat([v["boxes"] for v in targets])
#
# # Compute the classification cost. Contrary to the loss, we don't use the NLL,
# # but approximate it in 1 - proba[target class].
# # The 1 is a constant that doesn't change the matching, it can be ommitted.
# cost_class = -out_prob[:, tgt_ids]
#
# # Compute the L1 cost between boxes
# cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
#
# # Compute the giou cost betwen boxes
# out_bbox = box_cxcywh_to_xyxy(out_bbox)
# tgt_bbox = box_cxcywh_to_xyxy(tgt_bbox)
# cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
#
# indices = matcher(outputs, ty)
#
# loss_dict = criterion(outputs, ty)
# weight_dict = criterion.weight_dict
# losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

# %% PREPARE OUTPUTS
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
output_dir = Path(args.output_dir)
if args.resume:
    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

# %% TRAINING
print("Start training")
start_time = time.time()
for epoch in range(args.start_epoch, args.epochs):
    if args.distributed:
        sampler_train.set_epoch(epoch)
    train_stats = train_one_epoch_os(
        model, criterion, data_loader_train, optimizer, device, epoch,
        args.clip_max_norm)
    # train_stats = train_one_epoch(
    #     model, criterion, data_loader_train, optimizer, device, epoch,
    #     args.clip_max_norm)
    lr_scheduler.step()
    if args.output_dir:
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        # extra checkpoint before LR drop and every 100 epochs
        if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

    # test_stats, coco_evaluator = evaluate(
    #     model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
    # )
    coco_evaluator = None

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                 # **{f'test_{k}': v for k, v in test_stats.items()},
                 'epoch': epoch,
                 'n_parameters': n_parameters}

    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

        # for evaluation logs
        if coco_evaluator is not None:
            (output_dir / 'eval').mkdir(exist_ok=True)
            if "bbox" in coco_evaluator.coco_eval:
                filenames = ['latest.pth']
                if epoch % 50 == 0:
                    filenames.append(f'{epoch:03}.pth')
                for name in filenames:
                    torch.save(coco_evaluator.coco_eval["bbox"].eval,
                               output_dir / "eval" / name)

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))
