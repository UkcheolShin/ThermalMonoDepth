from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt

def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                         for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)

def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)

def opencv_rainbow_inv(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (0.60, 0.00, 1.00)),
        (0.400, (0.00, 0.00, 0.10)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (1.00, 1.00, 0.00)),
        (1.000, (1.00, 0.00, 0.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)

COLORMAPS = {'rainbow': opencv_rainbow(),
             'rainbow_inv' : plt.get_cmap('jet'), #opencv_rainbow_inv(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'magma_inv': high_res_colormap(cm.get_cmap('magma').reversed()),
             'bone': cm.get_cmap('bone', 10000),
             'bone_inv': cm.get_cmap('bone', 10000).reversed()}

def ind2rgb(im):
    cmap = plt.get_cmap('jet')
    im = im.cpu().squeeze().numpy()
    im = cmap(im)
    im = im[:,:,0:3]
    # put it from HWC to CHW format
    im = np.transpose(im, (2, 0, 1))
    return torch.from_numpy(im).float()
    
def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor[~np.isinf(tensor).type(torch.bool)].max()
        tensor[np.isinf(tensor).type(torch.bool)] = max_value
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        if( tensor.size(0) == 3) : 
            array = 0.45 + tensor.numpy()*0.225
        elif (tensor.size(0) == 2):
            array = tensor.numpy()

    return array

def tensor2array_thermal(tensor):
    tensor = tensor.detach().cpu()
    array = np.expand_dims(0.45 + tensor.detach().cpu().squeeze().numpy()*0.225, 0)
    return array

def save_checkpoint(save_path, dispnet_state, exp_pose_state, is_depth_best, is_pose_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'exp_pose']
    states = [dispnet_state, exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix, filename))

    if (is_depth_best&is_pose_best):
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix, filename),
                            save_path/'{}_both_model_best.pth.tar'.format(prefix))
    elif is_depth_best :
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix, filename),
                            save_path/'{}_disp_model_best.pth.tar'.format(prefix))
    elif is_pose_best : 
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix, filename),
                            save_path/'{}_pose_model_best.pth.tar'.format(prefix))        
