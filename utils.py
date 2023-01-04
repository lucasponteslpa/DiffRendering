import imageio
import json
import numpy as np

def get_video_reader(path):
    return imageio.get_reader(path, format='ffmpeg')

def get_image_from_video(video_reader, index):
    frame = video_reader.get_data(index)[:,:,::-1]
    return frame


def get_SMPL_parametes_from_json(path):
    f  = open(path)
    data = json.load(f)
    joints = np.array(data['joints'])
    translation = np.array(data['translation'])
    rotation = np.array(data['rotation'])
    f.close()
    return joints, translation, rotation

def close_reader(reader):
    reader.close()