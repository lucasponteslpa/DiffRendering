import argparse
from tqdm import tqdm
import os
import pathlib
import sys
import numpy as np
import torch
import imageio
import numpy as np
import cv2

from samples.torch import util

import nvdiffrast.torch as dr
from texture_tool.models.smpl_torch import SMPLModel

from render.obj import load_obj
from render.render import interpolate

from smpl_visualize import *

def fit_smpl(gctx,
             smpl_mesh_target,
             smpl_ref,
             poses,
             betas,
             trans,
             mvp_list,
             resolution,
             max_iter          = 5000,
             log_interval      = 10,
             display_interval  = None,
             display_res       = 512,
             out_dir           = None,
             log_fn            = None,
             mp4save_interval  = None,
             mp4save_fn        = None):

    # breakpoint()
    writer = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # if log_fn:
            # log_file = open(f'{out_dir}/{log_fn}', 'wt')
        if mp4save_interval != 0:
            writer = imageio.get_writer(f'{out_dir}/{mp4save_fn}', mode='I', fps=30, codec='libx264', bitrate='16M')
    else:
        mp4save_interval = None

    # vtx_pos_rand = np.random.uniform(-0.5, 0.5, size=vtxp.shape) + vtxp
    vtx_col_rand = np.random.uniform(0.0, 1.0, size=smpl_mesh_target.v_pos.shape)
    # vtx_pos_opt  = torch.tensor(vtx_pos_rand, dtype=torch.float32, device='cuda', requires_grad=True)
    vtx_col_opt  = torch.tensor(vtx_col_rand, dtype=torch.float32, device='cuda', requires_grad=True)

    # Adam optimizer for vertex position and color with a learning rate ramp.
    optimizer    = torch.optim.Adam([vtx_col_opt], lr=1e-2)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.01, 10**(-x*0.0005)))
    with tqdm(total=(max_iter)) as tq:
        for it in range(max_iter + 1):
            save_mp4 = mp4save_interval and (it % mp4save_interval == 0)
            for p, b, t in zip(poses,betas, trans):
                img_frames = []
                for n, m in enumerate(mvp_list):

                    color_target = gen_img_frame(gctx,smpl_mesh_target, smpl_ref, p, b, t, m, resolution)
                    color_opt = simple_render(gctx,
                                    m,
                                    smpl_mesh_target.v_pos,
                                    smpl_mesh_target.t_pos_idx.to(torch.int32),
                                    vtx_col_opt,
                                    torch.from_numpy(smpl_ref.faces.astype(np.int32)).cuda(),
                                    resolution)

                    # Compute loss and train.
                    loss = torch.mean((color_target - color_opt)**2) # L2 pixel loss.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    if save_mp4 and n < 2:
                        f = np.clip(np.rint(color_target[0].cpu().detach().numpy()*255),0,255).astype(np.uint8)
                        img_frames.append(f)
                        f = np.clip(np.rint(color_opt[0].cpu().detach().numpy()*255),0,255).astype(np.uint8)
                        img_frames.append(f)

                if save_mp4:
                    img_grid = make_grid(np.stack(img_frames))
                    writer.append_data(img_grid)

            tq.update()
    # Done.
    if writer is not None:
        writer.close()



def main():
    parser = argparse.ArgumentParser(description='Cube fit example')
    parser.add_argument('--outdir', help='specify output directory', default='smpl_out')
    parser.add_argument('--discontinuous', action='store_true', default=False)
    parser.add_argument('--resolution', type=int, default=32, required=False)
    parser.add_argument('--mp4save', action='store_true', default=False)
    parser.add_argument('--n_rand_poses', type=int, default=5, required=False)
    parser.add_argument('--n_interp_pose_size', type=int, default=100, required=False)
    parser.add_argument('--n_camera_poses', type=int, default=10, required=False)
    parser.add_argument('--display-interval', type=int, default=0)
    parser.add_argument('--mp4save-interval', type=int, default=100)
    parser.add_argument('--max-iter', type=int, default=5)
    args = parser.parse_args()

    writer = imageio.get_writer(f'{args.outdir}/smpl_rand_pose_1_cam.mp4', mode='I', fps=20, codec='libx264', bitrate='16M')

    smpl_f_mesh = load_obj('texture_tool/smpl_model/smpl_sample/SMPL/SMPL_female_default_resolution.obj',mtl_override='texture_tool/smpl_model/smpl_sample/SMPL/SMPL_female_default_resolution.mtl')
    glctx = dr.RasterizeCudaContext()

    smpl_vert, smpl_model, device = gen_smpl_vertices(gpu_id=[1])
    poses, betas, trans = gen_rand_smpl_param(device,
                                              n_base_points=args.n_rand_poses,
                                              interp_size=args.n_interp_pose_size)
    # Random rotation/translation matrix for optimization.
    mtx_list = []
    for i in range(args.n_camera_poses):

        r_rot = util.random_rotation_translation(0.25 + i*0.05)

        # Modelview and modelview + projection matrices.
        proj  = util.projection(x=0.4)
        r_mv  = np.matmul(util.translate(0, 0, -3.5), r_rot)
        r_mvp = np.matmul(proj, r_mv).astype(np.float32)
        mtx_list.append(r_mvp)

    fit_smpl(
        glctx,
        smpl_f_mesh,
        smpl_model,
        poses,
        betas,
        trans,
        mtx_list,
        args.resolution,
        max_iter=args.max_iter,
        log_interval=10,
        display_interval=args.display_interval,
        out_dir=args.outdir,
        log_fn='log.txt',
        mp4save_interval=args.mp4save_interval,
        mp4save_fn='progress.mp4'
    )
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()