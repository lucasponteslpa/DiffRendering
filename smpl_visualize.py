import argparse
import os
import pathlib
import sys
import numpy as np
import torch
import imageio
import numpy as np

from samples.torch import util

import nvdiffrast.torch as dr
from texture_tool.models.smpl_torch import SMPLModel

def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, resolution: int):
    pos_clip    = transform_pos(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    color, _    = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color

def make_grid(arr, ncols=2):
    n, height, width, nc = arr.shape
    nrows = n//ncols
    assert n == nrows*ncols
    return arr.reshape(nrows, ncols, height, width, nc).swapaxes(1,2).reshape(height*nrows, width*ncols, nc)

def gen_smpl_vertices(model_path='texture_tool/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl', gpu_id=[0]):
    if len(gpu_id) > 0 and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id[0])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)

    pose_size = 72
    beta_size = 10

    np.random.seed(9608)
    pose = torch.from_numpy((np.random.rand(pose_size) - 0.5) * 0.4)\
            .type(torch.float64).to(device)
    betas = torch.from_numpy((np.random.rand(beta_size) - 0.5) * 0.06) \
            .type(torch.float64).to(device)
    trans = torch.from_numpy(np.zeros(3)).type(torch.float64).to(device)
    # outmesh_path = './smpl_torch.obj'

    model = SMPLModel(device=device, model_path=model_path)
    result = model(betas, pose, trans)
    return result.float(), model, device

def main():
    parser = argparse.ArgumentParser(description='Cube fit example')
    parser.add_argument('--outdir', help='specify output directory', default='')
    parser.add_argument('--discontinuous', action='store_true', default=False)
    parser.add_argument('--resolution', type=int, default=32, required=False)
    args = parser.parse_args()

    glctx = dr.RasterizeCudaContext()

    smpl_vert, smpl_model, device = gen_smpl_vertices(gpu_id=[1])
    vtx_col_rand = np.random.uniform(0.0, 1.0, size=smpl_vert.shape)
    vtx_col_opt  = torch.tensor(vtx_col_rand, dtype=torch.float32, device=device, requires_grad=True)

    # Random rotation/translation matrix for optimization.
    r_rot = util.random_rotation_translation(0.25)

    # Smooth rotation for display.
    ang=0
    a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))

    # Modelview and modelview + projection matrices.
    proj  = util.projection(x=0.4)
    r_mv  = np.matmul(util.translate(0, 0, -3.5), r_rot)
    r_mvp = np.matmul(proj, r_mv).astype(np.float32)
    a_mv  = np.matmul(util.translate(0, 0, -3.5), a_rot)
    a_mvp = np.matmul(proj, a_mv).astype(np.float32)

    breakpoint()
    faces = torch.from_numpy(smpl_model.faces.astype(np.int32)).cuda()
    color     = render(glctx, r_mvp, smpl_vert, faces, vtx_col_opt, faces, args.resolution)
    # Set up logging.
    if args.outdir:
        ds = 'd' if args.discontinuous else 'c'
        out_dir = f'{args.outdir}/cube_{ds}_{args.resolution}'
        print (f'Saving results under {out_dir}')
    else:
        out_dir = None
        print ('No output directory specified, not saving log or images')


    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()