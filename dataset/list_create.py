import os
import random
import pandas as pd


def create_list(dir_path):   
    img_list = os.listdir(dir_path)  
    for idx, img_path in enumerate(img_list):
        img_list[idx] = os.path.join(dir_path, img_path)      
    return img_list


dir_1080p = create_list(os.path.join('images', '1080p'))
dir_960x960 = create_list(os.path.join('images', '960x960'))
dir_960x540 = create_list(os.path.join('images', '960x540'))
dir_540x540 = create_list(os.path.join('images', '540x540'))


train = []
train.extend(dir_1080p)
train.extend(dir_960x960)
train.extend(dir_960x540)
train.extend(dir_540x540)

val = random.sample(train, 100)


pd.DataFrame(train, columns=["images"]).to_csv('train.csv', index=False)
pd.DataFrame(val, columns=["images"]).to_csv('val.csv', index=False)