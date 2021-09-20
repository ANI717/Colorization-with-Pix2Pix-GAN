#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Hyperparameters
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2

IMG_SIZE = 512
IMG_CHANNELS = 3

BATCH_SIZE = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 50

L1_LAMBDA = 100

LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "checkpoint/gen.pth.tar"
CHECKPOINT_DISC = "checkpoint/disc.pth.tar"
RESULTS = 'results'


# Transformations
both_transform = A.Compose([
    A.Resize(height=512, width=512),
    ], additional_targets={"image0": "image"},)

transform_only_input = A.Compose([
    A.ColorJitter(p=0.2),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
    ToTensorV2(),])

transform_only_mask = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
    ToTensorV2(),])