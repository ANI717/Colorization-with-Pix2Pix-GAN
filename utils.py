#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import os
import torch
import config
from pathlib import Path
from torchvision.utils import save_image


# Load Checkpoint
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Save Checkpoint
def save_checkpoint(model, optimizer, filename='checkpoint/my_checkpoint.pth.tar'):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        }
    torch.save(checkpoint, filename)


# Save Sample Image
def save_sample_image(inputs, real, fake, epoch, step):
    Path(config.RESULTS).mkdir(parents=True, exist_ok=True)
    save_image(inputs, os.path.join(config.RESULTS, 'input.png'))
    save_image(real, os.path.join(config.RESULTS, 'real.png'))
    save_image(fake, os.path.join(config.RESULTS, f'fake_epoch{epoch}_step{step}.png'))