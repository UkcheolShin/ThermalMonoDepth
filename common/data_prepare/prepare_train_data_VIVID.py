import argparse
import scipy.misc
import numpy as np
from pebble import ProcessPool
import sys
from tqdm import tqdm
from path import Path
import cv2
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", metavar='DIR',
                    help='path to original dataset')
parser.add_argument("--with-depth", action='store_true',
                    help="If available, will store depth ground truth along with images, for validation")
parser.add_argument("--with-pose", action='store_true',
                    help="If available, will store pose ground truth along with images, for validation")
parser.add_argument("--no-train-gt", action='store_true',
                    help="If selected, will delete ground truth depth to save space")
parser.add_argument("--dump-root", type=str, default='dump', help="Where to dump the data")
parser.add_argument("--height", type=int, default=256, help="image height")
parser.add_argument("--width", type=int, default=320, help="image width")
parser.add_argument("--num-threads", type=int, default=4, help="number of threads to use")

args = parser.parse_args()

def dump_example(args, scene):
    scene_name = scene.split('/')[-1]
    scene_data = data_loader.collect_scenes(scene,)
    dump_dir   = args.dump_root/scene_data['rel_path']
    dump_dir.makedirs_p()
    dump_dir_ther      = dump_dir/"Thermal"
    dump_dir_rgb       = dump_dir/"RGB"
    dump_dir_depth_T   = dump_dir/"Depth_T"
    dump_dir_depth_RGB = dump_dir/"Depth_RGB"
    dump_dir_ther.makedirs_p()
    dump_dir_rgb.makedirs_p()
    dump_dir_depth_T.makedirs_p()
    dump_dir_depth_RGB.makedirs_p()

    # save intrinsic param
    intrinsics_T = scene_data['intrinsics_T']
    dump_cam_file = dump_dir/'cam_T.txt'
    np.savetxt(dump_cam_file, intrinsics_T) 

    intrinsics_RGB = scene_data['intrinsics_RGB']
    dump_cam_file = dump_dir/'cam_RGB.txt'
    np.savetxt(dump_cam_file, intrinsics_RGB) 

    extrinsics_T2RGB = scene_data['Tr_T2RGB']
    dump_cam_file = dump_dir/'Tr_T2RGB.txt'
    np.savetxt(dump_cam_file, extrinsics_T2RGB) 

    poses_T_file = dump_dir/'poses_T.txt'
    poses_RGB_file = dump_dir/'poses_RGB.txt'

    poses_T = []
    poses_RGB = []

    # save each files + pose + depth
    for sample in data_loader.get_scene_imgs(scene_data):
        frame_nb = sample["id"]
        cv2.imwrite(dump_dir_rgb/'{}.png'.format(frame_nb), sample['Img_RGB'])
        cv2.imwrite(dump_dir_ther/'{}.png'.format(frame_nb), sample['Img_Ther'])

        if "pose_T" in sample.keys():
            poses_T.append(sample["pose_T"].tolist())
            poses_RGB.append(sample["pose_RGB"].tolist())
        if "depth_T" in sample.keys():
            dump_depth_T_file = dump_dir_depth_T/'{}.npy'.format(frame_nb)
            np.save(dump_depth_T_file, sample["depth_T"])
            dump_depth_RGB_file = dump_dir_depth_RGB/'{}.npy'.format(frame_nb)
            np.save(dump_depth_RGB_file, sample["depth_RGB"])

    if len(poses_T) != 0:
        np.savetxt(poses_T_file, np.array(poses_T).reshape(-1, 12), fmt='%.6e')
        np.savetxt(poses_RGB_file, np.array(poses_RGB).reshape(-1, 12), fmt='%.6e')

    if len(dump_dir_rgb.files('*.png')) < 3:
        dump_dir.rmtree()

def extract_well_lit_images(args):
    tgt_dir   = args.dump_root/'indoor_robust_varying'
    dump_dir   = args.dump_root/'indoor_robust_varying_well_lit'   
    dump_dir.makedirs_p()
    dump_dir_ther      = dump_dir/"Thermal"
    dump_dir_rgb       = dump_dir/"RGB"
    dump_dir_depth_T   = dump_dir/"Depth_T"
    dump_dir_depth_RGB = dump_dir/"Depth_RGB"
    dump_dir_ther.makedirs_p()
    dump_dir_rgb.makedirs_p()
    dump_dir_depth_T.makedirs_p()
    dump_dir_depth_RGB.makedirs_p()

    # read well-lit image list
    img_list = np.genfromtxt('./common/data_prepare/well_lit_from_varying.txt').astype(int) # 

    for frame_nb in img_list :
        dump_img_T_file     = dump_dir_rgb/'{:06d}.png'.format(frame_nb)
        dump_img_RGB_file   = dump_dir_ther/'{:06d}.png'.format(frame_nb)        
        dump_depth_T_file   = dump_dir_depth_T/'{:06d}.npy'.format(frame_nb)
        dump_depth_RGB_file = dump_dir_depth_RGB/'{:06d}.npy'.format(frame_nb)

        shutil.copy(tgt_dir/"Thermal"/'{:06d}.png'.format(frame_nb), dump_img_T_file)
        shutil.copy(tgt_dir/"RGB"/'{:06d}.png'.format(frame_nb), dump_img_RGB_file)
        shutil.copy(tgt_dir/"Depth_T"/'{:06d}.npy'.format(frame_nb), dump_depth_T_file)
        shutil.copy(tgt_dir/"Depth_RGB"/'{:06d}.npy'.format(frame_nb), dump_depth_RGB_file)

def main():
    args.dump_root = Path(args.dump_root)
    args.dump_root.mkdir_p()

    global data_loader

    from VIVID_raw_loader import VIVIDRawLoader
    data_loader = VIVIDRawLoader(args.dataset_dir,
                                 img_height=args.height,
                                 img_width=args.width,
                                 get_depth=args.with_depth,
                                 get_pose=args.with_pose)

    n_scenes = len(data_loader.scenes)
    print('Found {} potential scenes'.format(n_scenes))
    print('Retrieving frames')
    if args.num_threads == 1:
        for scene in tqdm(data_loader.scenes):
            dump_example(args, scene)
    else:
        with ProcessPool(max_workers=args.num_threads) as pool:
            tasks = pool.map(dump_example, [args]*n_scenes, data_loader.scenes)
            try:
                for _ in tqdm(tasks.result(), total=n_scenes):
                    pass
            except KeyboardInterrupt as e:
                tasks.cancel()
                raise e

    print('Extracting well-lit image from varying illumination set')
    extract_well_lit_images(args)

    print('Generating train val lists')
    with open(args.dump_root / 'train_indoor.txt', 'w') as tf:
        for seq in data_loader.indoor_train_list :
            tf.write('{}\n'.format(seq))
    with open(args.dump_root / 'val_indoor.txt', 'w') as tf:
        for seq in data_loader.indoor_val_list :
            tf.write('{}\n'.format(seq))
    with open(args.dump_root / 'test_indoor.txt', 'w') as tf:
        for seq in data_loader.indoor_test_list :
            tf.write('{}\n'.format(seq))
    with open(args.dump_root / 'train_outdoor.txt', 'w') as tf:
        for seq in data_loader.outdoor_train_list :
            tf.write('{}\n'.format(seq))
    with open(args.dump_root / 'val_outdoor.txt', 'w') as tf:
        for seq in data_loader.outdoor_val_list :
            tf.write('{}\n'.format(seq))
    with open(args.dump_root / 'test_outdoor.txt', 'w') as tf:
        for seq in data_loader.outdoor_test_list :
            tf.write('{}\n'.format(seq))

    print('Done!')

if __name__ == '__main__':
    main()
