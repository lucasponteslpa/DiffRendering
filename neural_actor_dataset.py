import torch
import glob
import os
import numpy as np

from utils import *

class NeuralActorDataset(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self, video_path, smpl_param_path, fixed_params=True):
        super().__init__()

        self.fixed_params = fixed_params
        self.video_path = video_path
        self.video_names = glob.glob(self.video_path+'*.avi')
        self.video_reader = get_video_reader(video_path)

        self.smpl_params_path = smpl_param_path
        self.smpl_params_names = glob.glob(self.smpl_params_path+'*.json')

    def get_smpl_params_index_path(self, index):
        str_index = str(index)
        return os.path.join(self.smpl_params_path,(6-len(str_index))*'0'+str_index+'.json')

    def __len__(self):
        if self.fixed_params:
            return len(self.video_names)
        else:
            return len(self.smpl_params_names)

    def __getitem__(self, data_idx):
        if self.fixed_params:
            video_reader = get_video_reader(self.video_names[data_idx])

            joints, translation, rotation = get_SMPL_parametes_from_json(self.get_smpl_params_index_path(0))
            img_frame = get_image_from_video(video_reader,0)
        else:
            params_name = self.smpl_params_names[data_idx].split('/')
            index = int(params_name[-1].split('.')[0])

            joints, translation, rotation = get_SMPL_parametes_from_json(self.smpl_params_names[data_idx])
            img_frame = get_image_from_video(self.video_reader,index)

        out_dict = {'target_frame':torch.from_numpy(img_frame.astype(np.float32)/255.0),
                    'joints':torch.from_numpy(joints),
                    'translation': torch.from_numpy(translation),
                    'rotation': torch.from_numpy(rotation)}

        return out_dict