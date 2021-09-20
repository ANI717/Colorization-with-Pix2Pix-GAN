#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import modules
import os
import cv2
import numpy as np
from pathlib import Path


# Global variables
VIDEO_FILE = "video2.mp4"
COUNT_START = [588, 1175, 2349, 3533] # 1080p, 960x960, 960x540, 540x540


# Create directory if it doesn't exist
def create_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


# Cocatenate input and target image
def image_cat(image):
    gray = np.zeros_like(image)
    gray[:,:,0] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[:,:,1] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[:,:,2] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.concatenate((gray, image), axis = 1)


# Extract frame from video
def getFrame(cap, sec, count):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = cap.read()
    
    if hasFrames:
        if image.sum()/(image.shape[0]*image.shape[1]*image.shape[2]) > 30:
            
            # image shape: 1920×1080
            path_1080p = 'images/1080p/'
            create_dir(path_1080p)
            cv2.imwrite(os.path.join(path_1080p, str(count[0]) + ".jpg"), image_cat(image[:,:,:]))
            count[0] += 1
            
            # image shape: 960x960
            path_960 = 'images/960x960'
            create_dir(path_960)
            cv2.imwrite(os.path.join(path_960, str(count[1]) + ".jpg"), image_cat(image[:960,:960,:]))
            cv2.imwrite(os.path.join(path_960, str(count[1] + 1) + ".jpg"), image_cat(image[:960,960:,:]))
            count[1] += 2
            
            # image shape: 960x540
            path_960_540 = 'images/960x540'
            create_dir(path_960_540)
            cv2.imwrite(os.path.join(path_960_540, str(count[2]) + ".jpg"), image_cat(image[:540,:960,:]))
            cv2.imwrite(os.path.join(path_960_540, str(count[2] + 1) + ".jpg"), image_cat(image[:540,960:,:]))
            cv2.imwrite(os.path.join(path_960_540, str(count[2] + 2) + ".jpg"), image_cat(image[540:,:960,:]))
            cv2.imwrite(os.path.join(path_960_540, str(count[2] + 3) + ".jpg"), image_cat(image[540:,960:,:]))
            count[2] += 4
            
            # image shape: 540x540
            path_540 = 'images/540x540'
            create_dir(path_540)
            cv2.imwrite(os.path.join(path_540, str(count[3]) + ".jpg"), image_cat(image[:540,150:690,:]))
            cv2.imwrite(os.path.join(path_540, str(count[3] + 1) + ".jpg"), image_cat(image[:540,690:1230,:]))
            cv2.imwrite(os.path.join(path_540, str(count[3] + 2) + ".jpg"), image_cat(image[:540,1230:1770,:]))
            cv2.imwrite(os.path.join(path_540, str(count[3] + 3) + ".jpg"), image_cat(image[540:,150:690,:]))
            cv2.imwrite(os.path.join(path_540, str(count[3] + 4) + ".jpg"), image_cat(image[540:,690:1230,:]))
            cv2.imwrite(os.path.join(path_540, str(count[3] + 5) + ".jpg"), image_cat(image[540:,1230:1770,:]))
            count[3] += 6
            
    return hasFrames, count


def main():
    sec = 0
    frameRate = 0.5
    cap = cv2.VideoCapture(VIDEO_FILE)
    
    count = COUNT_START
    success = getFrame(cap, sec, count)
    while success:
        sec = sec + frameRate
        sec = round(sec, 2)
        success, count = getFrame(cap, sec, count)


if __name__ == '__main__':
    main()