# import argparse
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR

# import os
# from PIL import Image
# from copy import deepcopy
# from tqdm import tqdm
# import time

import torch
import numpy as np
from PIL import Image, ImageOps

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ROOT_DIR = './'
IMG_DIR = ROOT_DIR + 'symbol_images/'
IMG_SIZE = 45

from torchvision import transforms
IMG_TRANSFORM = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (1,))])

def pad_image(img, desired_size, fill=0):
    delta_w = desired_size - img.size[0]
    delta_h = desired_size - img.size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_img = ImageOps.expand(img, padding, fill)
    return new_img