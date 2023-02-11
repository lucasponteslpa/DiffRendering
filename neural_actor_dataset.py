import torch
import glob
import os
import numpy as np
import imageio

from utils import *

class NeuralActorDataset(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self, train_dir_path, device='cuda', fixed_params=True):
        super().__init__()
        self.device = device
        self.train_dir_path = train_dir_path
        self.fixed_params = fixed_params
        self.camera_frames_path = os.path.join(train_dir_path,'frames/*')
        self.camera_names = sorted(glob.glob(self.camera_frames_path)) # path to frames from each camera
        self.camera_frames_names = [sorted(glob.glob(c_dir+'/*')) for c_dir in self.camera_names]


        self.smpl_params_path = os.path.join(train_dir_path,'smpl_params')
        self.smpl_poses_name = glob.glob(self.smpl_params_path+'/smpl_poses.pt') # saved as pytorch tensor
        self.smpl_trans_name = glob.glob(self.smpl_params_path+'/smpl_trans.pt') # saved as pytorch tensor
        self.smpl_poses = torch.load(self.smpl_poses_name[0]).to(device)
        self.smpl_trans = torch.load(self.smpl_trans_name[0]).to(device)

        # self.smpl_pca_path = os.path.join(train_dir_path,'smpl_pca')
        # self.smpl_pca_names = glob.glob(self.smpl_pca_path+'*.pt') # saved as pytorch tensor

        self.mvp_path = os.path.join(train_dir_path,'mvp')
        self.mvp_names = glob.glob(self.mvp_path+'/mvp_list.pt') # saved as pytorch tensor
        self.mvp_list = torch.load(self.mvp_names[0]).to(device)

        self.camera_index = 0
        self.frame_index = 0

    def get_camera_len(self):
        return self.mvp_list.shape[0]

    def set_camera_idx(self, idx):
        self.camera_index = idx

    def reset_camera(self):
        self.camera_index = 0

    def next_camera(self):
        self.camera_index += 1

    def get_frame(self):
        frame = imageio.imread(self.camera_frames_names[self.camera_index][self.frame_index])
        out_dict = {'target_frame':torch.from_numpy(frame.astype(np.float32)/255.0).to(self.device),
                    'mvp': self.mvp_list[self.camera_index]}
        self.next_camera()
        return out_dict

    def __len__(self):
        return self.smpl_poses.shape[0]

    def __getitem__(self, data_idx):
        self.frame_index = data_idx
        out_dict = {'pose': self.smpl_poses[data_idx],
                    'translation': self.smpl_trans[data_idx],
                    }

        return out_dict