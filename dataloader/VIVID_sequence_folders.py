import torch
import torch.utils.data as data
import numpy as np

import math
import random
from imageio import imread
from path import Path

def load_as_float(path):
    return np.expand_dims(imread(path).astype(np.float32), axis=2) # HW -> HW1

class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/Thermal/0000000.png
        root/scene_1/Thermal/0000001.png
        ..
        root/scene_1/cam.txt
        root/scene_2/Thermal/0000000.png
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """
    def __init__(self, root, seed=None, train=True, sequence_length=3,\
                 tf_share=None, tf_input=None, tf_loss=None, \
                 scene_type='indoor', interval=1):
        np.random.seed(seed)
        random.seed(seed)

        self.root = Path(root)
        if scene_type == 'indoor': 
            folder_list_path = self.root/'train_indoor.txt' if train else self.root/'val_indoor.txt'
        elif scene_type == 'outdoor': 
            folder_list_path = self.root/'train_outdoor.txt' if train else self.root/'val_outdoor.txt'
        
        self.folders = [self.root/folder[:-1] for folder in open(folder_list_path)]
        self.tf_share = tf_share
        self.tf_input = tf_input
        self.tf_loss = tf_loss        
        self.crawl_folders(sequence_length, interval)

    def crawl_folders(self, sequence_length, interval):
        sequence_set = []
        demi_length = (sequence_length-1)//2 + interval - 1
        shifts = list(range(-demi_length, demi_length + 1))
        for i in range(1, 2*demi_length):
            shifts.pop(1)

        for folder in self.folders:
            imgs           = sorted((folder/"Thermal").files('*.png'))
            intrinsics     = np.genfromtxt(folder/'cam_T.txt').astype(np.float32).reshape((3, 3))

            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)

        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]

        # Read thermal images & GT depths
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        
        # Pre-process thermal images for network input & loss calculation
        imgs, intrinsics = self.tf_share([tgt_img] + ref_imgs, np.expand_dims(np.copy(sample['intrinsics']),axis=0))
        imgs_input, _  = self.tf_input(imgs, None)
        imgs_loss, _   = self.tf_loss(imgs, None)

        tgt_img_input = imgs_input[0]
        ref_imgs_input = imgs_input[1:]
        tgt_img_loss = imgs_loss[0]
        ref_imgs_loss = imgs_loss[1:]   

        return tgt_img_input, ref_imgs_input, tgt_img_loss, ref_imgs_loss, intrinsics.squeeze()

    def __len__(self):
        return len(self.samples)
