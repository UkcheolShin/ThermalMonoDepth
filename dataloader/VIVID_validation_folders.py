import torch
import torch.utils.data as data
import numpy as np

import math
import random
from imageio import imread
from path import Path

def load_as_float(path):
    return np.expand_dims(imread(path).astype(np.float32), axis=2) # HW -> HW1

class ValidationSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/Thermal/0000000.jpg
        root/scene_1/Depth/0000000.npy
        root/scene_1/Thermal/0000001.jpg
        root/scene_1/Depth/0000001.npy
        ..
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .

        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, tf_input=None, tf_loss=None, inference_folder = '', sequence_length=3, interval=1, scene_type='indoor'):
        self.root = Path(root)

        if inference_folder == '' : 
            if scene_type == 'indoor': 
                folder_list_path = self.root/'val_indoor.txt'
            elif scene_type == 'outdoor': 
                folder_list_path = self.root/'val_outdoor.txt'            
            self.folders = [self.root/folder[:-1] for folder in open(folder_list_path)]
        else:
            self.folders = [self.root/inference_folder]

        self.tf_input = tf_input
        self.tf_loss = tf_loss        

        self.crawl_folders(sequence_length, interval)

    def crawl_folders(self, sequence_length=3, interval=1):
        sequence_set = []
        demi_length = (sequence_length-1)//2 + interval - 1
        shifts = list(range(-demi_length, demi_length + 1))
        for i in range(1, 2*demi_length):
            shifts.pop(1)

        for folder in self.folders:      
            imgs = sorted((folder/"Thermal").files('*.png')) 
            for i in range(demi_length, len(imgs)-demi_length):
                depth = folder/"Depth_T"/(imgs[i].name[:-4] + '.npy')
                sample = {'tgt_img': imgs[i], 'tgt_depth': depth }
                sequence_set.append(sample)

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img  = load_as_float(sample['tgt_img'])
        depth    = np.load(sample['tgt_depth']).astype(np.float32)

        img_input, _ = self.tf_input([tgt_img], None)
        img_loss, _  = self.tf_loss([tgt_img], None) # used for visualization only 

        tgt_img_input  = img_input[0]
        tgt_img_loss   = img_loss[0]

        return tgt_img_input, tgt_img_loss, depth

    def __len__(self):
        return len(self.samples)


class ValidationSetPose(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/Thermal/0000000.jpg
        root/scene_1/Depth/0000000.npy
        root/scene_1/Thermal/0000001.jpg
        root/scene_1/Depth/0000001.npy
        ..
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .

        transform functions must take in a list a images and a numpy array which can be None
    """
    def __init__(self, root, tf_input=None, sequence_length=3, interval=1, scene_type='indoor'):
        self.root = Path(root)
        if scene_type == 'indoor': 
            scene_list_path = self.root/'val_indoor.txt'
        elif scene_type == 'outdoor': 
            scene_list_path = self.root/'val_outdoor.txt'

        self.folders = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.tf_input   = tf_input
        self.crawl_folders(sequence_length, step=1)

    def crawl_folders(self, sequence_length=3, step=1):
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        shift_range = np.array([step*i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1)

        for folder in self.folders:
            imgs = sorted((folder/"Thermal").files('*.png')) 
            poses  = np.genfromtxt(folder/'poses_T.txt').astype(np.float64).reshape(-1, 3, 4)

            # construct 5-snippet sequences
            tgt_indices = np.arange(demi_length, len(imgs) - demi_length).reshape(-1, 1)
            snippet_indices = shift_range + tgt_indices
            
            for indices in snippet_indices :
                sample = {'imgs' : [], 'poses' : []}
                for i in indices :
                    sample['imgs'].append(imgs[i])
                    sample['poses'].append(poses[i])
                sequence_set.append(sample)

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [load_as_float(img) for img in sample['imgs']]
        imgs, _ = self.tf_input(imgs, None)

        poses = np.stack([pose for pose in sample['poses']])
        first_pose = poses[0]
        poses[:,:,-1] -= first_pose[:,-1]
        compensated_poses = np.linalg.inv(first_pose[:,:3]) @ poses

        return imgs, compensated_poses

    def __len__(self):
        return len(self.samples)
