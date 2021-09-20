#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from skimage import io
import albumentations as A
from albumentations.pytorch import ToTensorV2
from generator import Generator


# Global Variables
IMG_PATH = 'sample/sample6.jpg'
OUTPUT_IMG_PATH = 'sample/output_sample6.jpg'

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CHECKPOINT_GEN = "checkpoint/gen.pth.tar"


# Transformation
transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
    ToTensorV2(),])

# Load Image
image = io.imread(IMG_PATH)
height, width, channel = image.shape
image = transform(image=image)["image"]
image = image.unsqueeze(0)

# Load Model
model_gen = Generator(in_channels=3).to(DEVICE)
model_gen.load_state_dict(torch.load(CHECKPOINT_GEN, map_location=DEVICE)["state_dict"])
model_gen.eval()

# Generate Output
output_image = model_gen(image.to(DEVICE))
output_image = F.interpolate(output_image, size=(height, width), mode='bicubic', align_corners=False)
output_image = torchvision.utils.make_grid(output_image, normalize=True)
save_image(output_image, OUTPUT_IMG_PATH)