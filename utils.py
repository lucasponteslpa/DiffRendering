import imageio
import json
import numpy as np

def get_video_reader(path):
    return imageio.get_reader(path, format='ffmpeg')

def get_image_from_video(video_reader, index, invert=False):
    if invert:
        frame = video_reader.get_data(index)[:,:,::-1]
    else:
        frame = video_reader.get_data(index)
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

def load_matrix(path):
    lines = [[float(w) for w in line.strip().split()] for line in open(path)]
    if len(lines[0]) == 2:
        lines = lines[1:]
    if len(lines[-1]) == 2:
        lines = lines[:-1]
    return np.array(lines).astype(np.float32)

def load_intrinsics(filepath, resized_width=None, invert_y=False):
    try:
        intrinsics = load_matrix(filepath)
        if intrinsics.shape[0] == 3 and intrinsics.shape[1] == 3:
            _intrinsics = np.zeros((4, 4), np.float32)
            _intrinsics[:3, :3] = intrinsics
            _intrinsics[3, 3] = 1
            intrinsics = _intrinsics
        if intrinsics.shape[0] == 1 and intrinsics.shape[1] == 16:
            intrinsics = intrinsics.reshape(4, 4)
        return intrinsics
    except ValueError:
        pass

    # Get camera intrinsics
    with open(filepath, 'r') as file:

        f, cx, cy, _ = map(float, file.readline().split())
    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices

    full_intrinsic = np.array([[fx, 0., cx,  0.],
                               [0., fy, cy,  0.],
                               [0., 0.,  1., 0.],
                               [0,  0.,  0., 1.]])

    return full_intrinsic