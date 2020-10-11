from torch.utils.data import Dataset
import numpy as np
import os
import glob
from PIL import Image

INSTRE_PATH = "/datadrive/crr/datasets/instre"
TRAIN_PATH = INSTRE_PATH + "/INSTRE-S-TRAIN"
TEST_PATH = INSTRE_PATH + "/INSTRE-S-TEST"


def read_instre_box(box_file):
    """
    Returns box with format [x0, y0, w, h]
    """

    with open(box_file, "r") as f:
        box = f.read().strip("\n").split(" ")
        box = [int(c) for c in box]
    return box


# %%
class_dirs = glob.glob(os.path.join(TRAIN_PATH, "*/"))

c0 = class_dirs[0]
sample0 = "002"

c0_s0 = glob.glob(os.path.join(c0, sample0 + ".*"))

s0_img = np.asarray(Image.open(c0_s0[1]))
s0_box = read_instre_box(c0_s0[0])

s0_img
s0_box

#%%
class OSDDataset(Dataset):
    def __init__(self, class_dirs):
        self.class_dirs = class_dirs