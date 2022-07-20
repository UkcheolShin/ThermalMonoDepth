from __future__ import division
import numpy as np
from path import Path
import cv2

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

class VIVIDRawLoader(object):
    def __init__(self,
                 dataset_dir,
                 img_height=256, 
                 img_width=320, 
                 min_speed=0.15, # m/sec
                 get_depth=False,
                 get_pose=False):
        dir_path = Path(__file__).realpath().dirname()

        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.folder_list = ['RGB', 'Thermal', 'Warped_Depth']
        self.date_list = ['indoor_aggresive_global', 'indoor_robust_dark', 'indoor_robust_varying', 'indoor_unstable_local', 'outdoor_robust_night1', 'indoor_aggresive_dark', 'indoor_robust_global', 'indoor_unstable_dark', 'outdoor_robust_day1', 'outdoor_robust_night2', 'indoor_aggresive_local', 'indoor_robust_local', 'indoor_unstable_global', 'outdoor_robust_day2']
        self.indoor_train_list = ['indoor_aggresive_global', 'indoor_unstable_local', 'indoor_robust_global', 'indoor_robust_local', 'indoor_unstable_global']
        self.indoor_val_list = ['indoor_robust_dark', 'indoor_aggresive_local']
        self.indoor_test_list = ['indoor_robust_dark', 'indoor_robust_varying', 'indoor_aggresive_dark', 'indoor_unstable_dark', 'indoor_aggresive_local']
        self.outdoor_train_list = ['outdoor_robust_day1', 'outdoor_robust_day2']
        self.outdoor_val_list = ['outdoor_robust_night1']
        self.outdoor_test_list = ['outdoor_robust_night1', 'outdoor_robust_night2']
        self.min_speed = min_speed
        self.get_depth = get_depth
        self.get_pose = get_pose
        self.collect_train_folders()

    # collect dataset sequence path.
    def collect_train_folders(self):
        self.scenes = []
        for date in self.date_list:
            self.scenes.append((self.dataset_dir/date))

    # collect subset from target 
    def collect_scenes(self, drive):
        scene_data = {'dir': drive, 'speed': [], 'frame_id': [], 'pose_T':[], 'pose_RGB':[], 'rel_path': drive.name}
        ther2rgb = self.read_raw_calib_file(drive.parent/'calibration'/'calib_ther_to_rgb.yaml')
#            event2rgb = self.read_raw_calib_file(drive.parent/'calibration'/'calib_event_to_rgb.yaml')
#            lidar2rgb = self.read_raw_calib_file(drive.parent/'calibration'/'calib_lidar_to_rgb.yaml')
#            rgb2imu = self.read_raw_calib_file(drive.parent/'calibration'/'calib_rgb_to_imu.yaml')
#            vicon2rgb = self.read_raw_calib_file(drive.parent/'calibration'/'calib_vicon_to_rgb.yaml')
#            ther2rgb_mat = transform_from_rot_trans(ther2rgb['R'], ther2rgb['T'])

        # read pose in target coord
        TgtPoseVec_T = np.genfromtxt(drive/'poses_thermal.txt') # len x 12 (3x4 matrix)
        TgtPoseMat_T = TgtPoseVec_T.reshape(-1,3,4)

        TgtPoseVec_RGB = np.genfromtxt(drive/'poses_RGB.txt') # len x 12 (3x4 matrix)
        TgtPoseMat_RGB = TgtPoseVec_RGB.reshape(-1,3,4)

        # read avg speed in target coor
        TgtAvgVelo_T = np.genfromtxt(drive/'avg_velocity_thermal.txt') # len x 1

        for n in range(len(TgtAvgVelo_T)):
            scene_data['pose_T'].append(TgtPoseMat_T[n])
            scene_data['pose_RGB'].append(TgtPoseMat_RGB[n])
            scene_data['frame_id'].append('{:06d}'.format(n+1)) # files are start from 1, but python indexing start from 0
            scene_data['speed'].append(TgtAvgVelo_T[n])

        sample_T = self.load_image(scene_data, 1, 'T')
        if sample_T is None:
            return []
        sample_RGB = self.load_image(scene_data, 1, 'RGB')
        if sample_RGB is None:
            return []

        scene_data['P_rect_T'] = self.get_P_rect(scene_data, sample_T[1], sample_T[2], 'T')
        scene_data['P_rect_RGB'] = self.get_P_rect(scene_data, sample_RGB[1], sample_RGB[2], 'RGB')
        scene_data['Tr_T2RGB'] = self.get_T_T2RGB(scene_data)
        scene_data['intrinsics_T'] = scene_data['P_rect_T'][:,:3]
        scene_data['intrinsics_RGB'] = scene_data['P_rect_RGB'][:,:3]
        scene_data['FolderList'] = self.folder_list

        return scene_data

    # sampling according to its moving velocity
    def get_scene_imgs(self, scene_data):
        def construct_sample(scene_data, i, frame_id):
            sample = {"id":frame_id}
            Img_RGB, Img_Ther = self.load_All_image(scene_data, i)
            sample['Img_RGB'] = Img_RGB
            sample['Img_Ther'] = Img_Ther

            if self.get_depth:
                sample['depth_T'] = self.load_Depth(scene_data, i, 'T')
                sample['depth_RGB'] = self.load_Depth(scene_data, i, 'RGB')
            if self.get_pose:
                sample['pose_T'] = scene_data['pose_T'][i]
                sample['pose_RGB'] = scene_data['pose_RGB'][i]

            return sample

        cum_speed = np.zeros(1)
        for i, speed in enumerate(scene_data['speed']):
            cum_speed += speed
            speed_mag = np.linalg.norm(cum_speed)
            if speed_mag > self.min_speed:
                frame_id = scene_data['frame_id'][i]
                yield construct_sample(scene_data, i, frame_id)
                cum_speed *= 0

    def get_P_rect(self, scene_data, zoom_x, zoom_y, Dtype):
        calib_file = scene_data['dir'].parent/'calibration'/'calib_ther_to_rgb.yaml'

        filedata = self.read_raw_calib_file(calib_file)
        if Dtype == 'T':
            K_ = np.reshape(filedata['K_Thermal'], (3, 3))
        elif Dtype == 'RGB':
            K_ = np.reshape(filedata['K_RGB'], (3, 3))            
        P_rect = K_@np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        P_rect[0] *= zoom_x
        P_rect[1] *= zoom_y
        return P_rect

    def get_T_T2RGB(self, scene_data):
        calib_file = scene_data['dir'].parent/'calibration'/'calib_ther_to_rgb.yaml'
        filedata = self.read_raw_calib_file(calib_file)
        R = filedata['R']
        T = filedata['T']
        T_T2RGB = transform_from_rot_trans(R,T)
        return T_T2RGB

    def load_image(self, scene_data, tgt_idx, Dtype):
        if Dtype == 'T':
            img_file = scene_data['dir']/'Thermal'/'data'/scene_data['frame_id'][tgt_idx]+'.png' 
        elif Dtype == 'RGB':
            img_file = scene_data['dir']/'RGB'/'data'/scene_data['frame_id'][tgt_idx]+'.png' 

        if not img_file.isfile():
            return None
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        zoom_y = self.img_height/img.shape[0]
        zoom_x = self.img_width/img.shape[1]
        img = cv2.resize(img, (self.img_width, self.img_height)) # shape is width,height order
        return img, zoom_x, zoom_y

    def load_All_image(self, scene_data, tgt_idx):
        img_file0 = scene_data['dir']/'RGB'/'data'/scene_data['frame_id'][tgt_idx]+'.png'      # uint8, 3ch
        img_file1 = scene_data['dir']/'Thermal'/'data'/scene_data['frame_id'][tgt_idx]+'.png'  # uint16, 1ch
        if not img_file1.isfile():
            return None

        img0 = cv2.imread(img_file0, cv2.IMREAD_UNCHANGED) # RGB
        img0 = cv2.resize(img0, (self.img_width, self.img_height)) # shape is width,height order

        img1 = cv2.imread(img_file1, cv2.IMREAD_UNCHANGED)
        img1 = cv2.resize(img1, (self.img_width, self.img_height)) # shape is width,height order

        return img0, img1

    def load_Depth(self, scene_data, tgt_idx, Dtype):
        if Dtype == 'T':
            depth_file = scene_data['dir']/'Warped_Depth'/'data_THERMAL'/scene_data['frame_id'][tgt_idx]+'.npy' 
        if Dtype == 'RGB':
            depth_file = scene_data['dir']/'Warped_Depth'/'data_RGB'/scene_data['frame_id'][tgt_idx]+'.npy' 
        if not depth_file.isfile():
            return None
        depth = np.load(depth_file)

        return depth

    def read_raw_calib_file(self, filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                        data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                        pass
        return data
