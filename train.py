#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from pathlib import Path
from tqdm import tqdm

import config
from dataset import ANI717Dataset
from generator import Generator
from discriminator import Discriminator
from utils import save_checkpoint, load_checkpoint, save_sample_image


# Main Method
def main():
    
    # Load Data
    train_dataset = ANI717Dataset('dataset/train.csv','dataset')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_dataset = ANI717Dataset('dataset/val.csv','dataset')
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    fixed_input, fixed_target = next(iter(val_loader))
    
    # Initialize Model
    model_gen = Generator(in_channels=3).to(config.DEVICE)
    model_disc = Discriminator(in_channels=3).to(config.DEVICE)
    
    # Initialize Optimizer and Loss
    optimizer_gen = optim.Adam(model_gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_disc = optim.Adam(model_disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    criterion_BCE = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    # Load Checkpoint
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, model_gen, optimizer_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, model_disc, optimizer_disc, config.LEARNING_RATE)
    
    # Test Block
    print(next(iter(train_dataset))[0].shape)
    import sys
    sys.exit()
    
    step = 0
    
    # Training
    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(train_loader, leave=True)
        for batch_idx, (x, y) in enumerate(loop):
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            
            # Train Discriminator
            with torch.cuda.amp.autocast():
                y_fake = model_gen(x)
                D_real = model_disc(x, y)
                D_real_loss = criterion_BCE(D_real, torch.ones_like(D_real))
                D_fake = model_disc(x, y_fake.detach())
                D_fake_loss = criterion_BCE(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2
            
            model_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(optimizer_disc)
            d_scaler.update()
            
            # Train Generator
            with torch.cuda.amp.autocast():
                D_fake = model_disc(x, y_fake)
                G_fake_loss = criterion_BCE(D_fake, torch.ones_like(D_fake))
                L1 = criterion_L1(y_fake, y) * config.L1_LAMBDA
                G_loss = G_fake_loss + L1
            
            model_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(optimizer_gen)
            g_scaler.update()
            
            # Print losses occasionally and print to tensorboard
            if batch_idx % 200 == 0:
                loop.set_postfix(
                    D_real=torch.sigmoid(D_real).mean().item(),
                    D_fake=torch.sigmoid(D_fake).mean().item(),
                )
                
                with torch.no_grad():
                    fake_target = model_gen(fixed_input.to(config.DEVICE))
                    
                    # take out (up to) 32 examples
                    img_grid_input = torchvision.utils.make_grid(fixed_input[:8], nrow=4, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(fixed_target[:8], nrow=4, normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake_target[:8], nrow=4, normalize=True)
                    
                    # Save Sample Generated Images
                    save_sample_image(img_grid_input, img_grid_real, img_grid_fake, epoch, step)
                    
                step += 1
        
        # Save Model in Every Epoch
        if config.SAVE_MODEL:
            Path(config.CHECKPOINT_GEN.split('/')[0]).mkdir(parents=True, exist_ok=True)
            save_checkpoint(model_gen, optimizer_gen, config.CHECKPOINT_GEN)
            
            Path(config.CHECKPOINT_DISC.split('/')[0]).mkdir(parents=True, exist_ok=True)
            save_checkpoint(model_disc, optimizer_disc, config.CHECKPOINT_DISC)


if __name__ == '__main__':
    main()