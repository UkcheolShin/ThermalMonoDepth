import torch
from skimage.transform import resize as imresize
from imageio import imread
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import time

import sys
sys.path.append('./common/')
import models

parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height") # 256 (kitti)
parser.add_argument("--img-width", default=320, type=int, help="Image width")   # 832 (kitti)
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--output-dir", default=None, required=True, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument('--resnet-layers', required=True, type=int, default=18, choices=[18, 50], help='depth network architecture.')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
def load_tensor_image(filename, args):
    img = np.expand_dims(imread(filename).astype(np.float32), axis=2)
    h,w,_ = img.shape
    if (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    img = (torch.from_numpy(img).float() / 2**14)
    tensor_img = ((img.unsqueeze(0)-0.45)/0.225).to(device)
    return tensor_img

@torch.no_grad()
def main():
    args = parser.parse_args()

    # load models
    disp_pose_net = models.DispPoseResNet(args.resnet_layers, False, num_channel=1).to(device)
    disp_net = disp_pose_net.DispResNet

    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)

    # read file list
    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = list(f.read().splitlines())
    else:
        test_files=sorted((dataset_dir+'Thermal').files('*.png'))

    print('{} files to test'.format(len(test_files)))
  
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    test_disp_avg = 0
    test_disp_std = 0
    test_depth_avg = 0
    test_depth_std = 0

    avg_time = 0
    for j in tqdm(range(len(test_files))):
        tgt_img = load_tensor_image(test_files[j], args)

        # compute speed
        torch.cuda.synchronize()
        t_start = time.time()

        output = disp_net(tgt_img)

        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        
        avg_time += elapsed_time

        pred_disp = output.squeeze().cpu().numpy()

        if j == 0:
            predictions = np.zeros((len(test_files), *pred_disp.shape))
        predictions[j] = 1/pred_disp

        test_disp_avg += pred_disp.mean()
        test_disp_std += pred_disp.std()
        test_depth_avg += predictions.mean()
        test_depth_std += predictions.std()
        
    np.save(output_dir/'predictions.npy', predictions)

    avg_time /= len(test_files)
    print('Avg Time: ', avg_time, ' seconds.')
    print('Avg Speed: ', 1.0 / avg_time, ' fps')

    print('Avg disp : {0:0.3f}, std disp : {1:0.5f}'.format(test_disp_avg/len(test_files), test_disp_std/len(test_files)))
    print('Avg depth: {0:0.3f}, std depth: {1:0.5f}'.format(test_depth_avg/len(test_files), test_depth_std/len(test_files)))


if __name__ == '__main__':
    main()
