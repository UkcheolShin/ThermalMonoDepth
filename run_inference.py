import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from imageio import imsave
import numpy as np
from path import Path
import argparse

import sys
sys.path.append('./common/')

import models 
import utils.custom_transforms as custom_transforms
from utils.utils import tensor2array

import matplotlib as mpl
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument("--sequence", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")
parser.add_argument("--img-exts", default='jpg', choices=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

parser.add_argument('--resnet-layers', required=True, type=int, default=18, choices=[18, 50],
                    help='depth network architecture.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')

parser.add_argument('--interval', type=int, help='Interval of sequence', metavar='N', default=1)
parser.add_argument('--sequence_length', type=int, help='Length of sequence', metavar='N', default=3)
parser.add_argument('--pretrained-model', dest='pretrained_model', default=None, metavar='PATH', help='path to pre-trained model')
parser.add_argument('--scene_type', type=str, choices=['indoor', 'outdoor'], default='indoor', required=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def depth_visualizer(inv_depth):
    """
    Args:
        data (HxW): depth data
    Returns:
        vis_data (HxWx3): depth visualization (RGB)
    """
    inv_depth = inv_depth.squeeze().detach().cpu()
    vmax = np.percentile(inv_depth, 98)
    normalizer = mpl.colors.Normalize(vmin=inv_depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    vis_data = (mapper.to_rgba(inv_depth)[:, :, :3] * 255).astype(np.uint8)
    return vis_data

@torch.no_grad()
def main():
    args = parser.parse_args()

    ArrToTen      = custom_transforms.ArrayToTensor(max_value=2**14)
    ArrToTen4Loss = custom_transforms.ArrayToTensorWithLocalWindow()
    TenThrRearran = custom_transforms.TensorThermalRearrange(bin_num=30, CLAHE_clip=2)
    normalize     = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])

    transform4input  = custom_transforms.Compose([ArrToTen, normalize])
    transform4loss   = custom_transforms.Compose([ArrToTen4Loss, TenThrRearran, normalize])

    from dataloader.VIVID_validation_folders import ValidationSet
    val_set = ValidationSet(
        args.data,
        tf_input     = transform4input,
        tf_loss      = transform4loss,
        sequence_length = args.sequence_length,
        interval     = args.interval,
        scene_type   = args.scene_type,        
        inference_folder=args.sequence,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # 1. Load models
    # create model
    print("=> creating model")
    disp_pose_net = models.DispPoseResNet(args.resnet_layers, False, num_channel=1).to(device)
    disp_net = disp_pose_net.DispResNet

    # load parameters
    print("=> using pre-trained weights for DispResNet")
    weights = torch.load(args.pretrained_model)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    # 2. Load dataset
    output_dir = Path(args.output_dir+'/'+args.sequence)
    output_dir.makedirs_p()
    (output_dir/'thr_img').makedirs_p()
    (output_dir/'thr_invdepth').makedirs_p()

    disp_net.eval()

    for idx, (tgt_img, tgt_img_vis, depth_gt) in enumerate(val_loader):

        # original validate_with_gt param
        tgt_img   = tgt_img.to(device)
        depth_gt  = depth_gt.to(device)

        # compute output
        output_disp = disp_net(tgt_img)

        tgt_img_vis        = (255*(0.45 + tgt_img_vis.squeeze().detach().cpu().numpy()*0.225)).astype(np.uint8)
        tgt_disp           = depth_visualizer(output_disp)

        # Save images
        file_name = '{:06d}'.format(idx)
        imsave(output_dir/'thr_img'/'{}.{}'.format(file_name, args.img_exts), tgt_img_vis)
        imsave(output_dir/'thr_invdepth'/'{}.{}'.format(file_name, args.img_exts), tgt_disp)

if __name__ == '__main__':
    main()
