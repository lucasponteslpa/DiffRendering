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

from mssim import SSIM
from samples.torch import util

import nvdiffrast.torch as dr
from texture_tool.models.smpl_torch import SMPLModel

from render.obj import load_obj
from render.render import interpolate

from mlp_models.mlp_models import SimpleColorMLP

from render.texture import texture2d_mip

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
             fit_mode          = None,
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
    if fit_mode == 'tex':
        mat_shape = smpl_mesh_target.material['kd'].data.shape
        mat_shape_1 = (mat_shape[1]//4)//2
        mat_shape_2 = (mat_shape[2]//4)//2
        opt_shape = (mat_shape[0], mat_shape_1, mat_shape_2, mat_shape[3])
        # vtx_col_rand = np.random.uniform(0.0, 1.0, size=(mat_shape[0], mat_shape_1, mat_shape_2, mat_shape[3]))
        # vtx_pos_opt  = torch.tensor(vtx_pos_rand, dtype=torch.float32, device='cuda', requires_grad=True)
        # vtx_col_opt  = torch.tensor(vtx_col_rand, dtype=torch.float32, device='cuda', requires_grad=True)
        vtx_col_opt=torch.full(opt_shape, 0.2, device='cuda', requires_grad=True)
    elif fit_mode == 'mlp':
        model = SimpleColorMLP().to('cuda')
        model.train()
        canonical_v_pos = torch.clone(smpl_mesh_target.v_pos).to('cuda')
    elif fit_mode == 'tex_mlp' or fit_mode == 'tex_mlp2':
        mat_shape = smpl_mesh_target.material['kd'].data.shape
        mat_shape_1 = mat_shape[1]//8
        mat_shape_2 = mat_shape[2]//8
        coords = torch.linspace(-1.0,1.0,mat_shape_1,dtype=torch.float32, device='cuda')
        x_grid, y_grid = torch.meshgrid(coords, coords, indexing='ij')
        x_coords = x_grid.reshape(-1).unsqueeze(-1)
        y_coords = y_grid.reshape(-1).unsqueeze(-1)
        xy_coords = torch.cat((x_coords,y_coords), -1)
        model = SimpleColorMLP(pos_enc_len=6, in_channels=2, n_mlp_layers=8).to('cuda')

    else:
        vtx_col_rand = np.random.uniform(0.0, 1.0, size=smpl_mesh_target.v_pos.shape)
        # vtx_pos_opt  = torch.tensor(vtx_pos_rand, dtype=torch.float32, device='cuda', requires_grad=True)
        vtx_col_opt  = torch.tensor(vtx_col_rand, dtype=torch.float32, device='cuda', requires_grad=True)

    # Adam optimizer for vertex position and color with a learning rate ramp.
    if not fit_mode=='mlp' and not fit_mode=='tex_mlp':
        optimizer    = torch.optim.Adam([vtx_col_opt], lr=1e-2)
    else:
        optimizer    = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.01, 10**(-x*0.0005)))
    ssim = SSIM(data_range=1.0)
    with tqdm(total=poses.shape[0]*len(mvp_list)-1) as tq:
        for it in range(max_iter):
            print("Epoch {}/{}".format(it+1, max_iter))
            save_mp4 = mp4save_interval and (it % mp4save_interval == 0)
            for p, b, t in zip(poses,betas, trans):
                img_frames = []
                for n, m in enumerate(mvp_list):
                    if fit_mode =='tex_mlp2':
                        color_target = torch.nn.functional.interpolate(smpl_mesh_target.material['kd'].data.permute(0,3,1,2), scale_factor=0.25,mode='bicubic')
                        color_target = color_target.permute(0,2,3,1)
                    else:
                        color_target = gen_img_frame(gctx,smpl_mesh_target, smpl_ref, p, b, t, m, resolution)
                    if fit_mode=='tex':
                        mips = [vtx_col_opt]
                        while mips[-1].shape[1] > 1 and mips[-1].shape[2] > 1:
                            mips += [texture2d_mip.apply(mips[-1])]
                        color_opt = texture_render(gctx,
                                        m,
                                        smpl_mesh_target.v_pos,
                                        smpl_mesh_target.t_pos_idx.to(torch.int32),
                                        smpl_mesh_target.v_tex,
                                        smpl_mesh_target.t_tex_idx.to(torch.int32),
                                        vtx_col_opt,
                                        resolution,
                                        mip=mips[1:])
                    elif fit_mode=='mlp':
                        vtx_col_opt = model(canonical_v_pos)
                        color_opt = simple_render(gctx,
                                        m,
                                        smpl_mesh_target.v_pos,
                                        smpl_mesh_target.t_pos_idx.to(torch.int32),
                                        vtx_col_opt,
                                        torch.from_numpy(smpl_ref.faces.astype(np.int32)).cuda(),
                                        resolution)
                    elif fit_mode=='tex_mlp':
                        vtx_col_opt = model(xy_coords)
                        vtx_col_opt = vtx_col_opt.reshape(1,mat_shape_1, mat_shape_2, 3)
                        mips = [vtx_col_opt]
                        while mips[-1].shape[1] > 1 and mips[-1].shape[2] > 1:
                            mips += [texture2d_mip.apply(mips[-1])]
                        color_opt = texture_render(gctx,
                                        m,
                                        smpl_mesh_target.v_pos,
                                        smpl_mesh_target.t_pos_idx.to(torch.int32),
                                        smpl_mesh_target.v_tex,
                                        smpl_mesh_target.t_tex_idx.to(torch.int32),
                                        vtx_col_opt,
                                        resolution,
                                        mip=mips[1:])
                    elif fit_mode=='tex_mlp2':
                        vtx_col_opt = model(xy_coords)
                        color_opt = vtx_col_opt.reshape(1,mat_shape_1, mat_shape_2, 3)

                    else:
                        color_opt = simple_render(gctx,
                                        m,
                                        smpl_mesh_target.v_pos,
                                        smpl_mesh_target.t_pos_idx.to(torch.int32),
                                        vtx_col_opt,
                                        torch.from_numpy(smpl_ref.faces.astype(np.int32)).cuda(),
                                        resolution)

                    # Compute loss and train.
                    loss = torch.mean((color_target - color_opt)**2) + (1.0 - ssim(color_target.permute(0,3,1,2),color_opt.permute(0,3,1,2)))/2.0 # L2 pixel loss.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    tq.set_postfix({'mse':loss.item()})
                    tq.update()

                    if save_mp4 and n < 2:
                        f = np.clip(np.rint(color_target[0].cpu().detach().numpy()*255),0,255).astype(np.uint8)
                        img_frames.append(f)
                        f = np.clip(np.rint(color_opt[0].cpu().detach().numpy()*255),0,255).astype(np.uint8)
                        img_frames.append(f)

                if save_mp4:
                    img_grid = make_grid(np.stack(img_frames))
                    writer.append_data(img_grid)
            tq.refresh()
            tq.reset()
    # Done.
    if fit_mode=='tex' or fit_mode=='tex_mlp':
        if fit_mode=='tex_mlp':
            vtx_col_opt = model(xy_coords)
            vtx_col_opt = vtx_col_opt.reshape(1,mat_shape_1, mat_shape_2,3)
        cv2.imwrite(f'{out_dir}/final_tex.png',vtx_col_opt.cpu().detach().numpy()[0,:,:,::-1]*255)
        color_opt = texture_render(gctx,
                                        mvp_list[0],
                                        smpl_mesh_target.v_pos,
                                        smpl_mesh_target.t_pos_idx.to(torch.int32),
                                        smpl_mesh_target.v_tex,
                                        smpl_mesh_target.t_tex_idx.to(torch.int32),
                                        vtx_col_opt,
                                        resolution)
        cv2.imwrite(f'{out_dir}/final_col_cam0.png',color_opt.cpu().detach().numpy()[0,:,:,::-1]*255)
    else:
        if fit_mode=='mlp':
            vtx_col_opt = model(canonical_v_pos)
        color_opt = simple_render(gctx,
                                  mvp_list[0],
                                  smpl_mesh_target.v_pos,
                                  smpl_mesh_target.t_pos_idx.to(torch.int32),
                                  vtx_col_opt,
                                  torch.from_numpy(smpl_ref.faces.astype(np.int32)).cuda(),
                                  resolution*2)
        cv2.imwrite(f'{out_dir}/final_col_cam0.png',color_opt.cpu().detach().numpy()[0,:,:,::-1]*255)
    if writer is not None:
        writer.close()



def main():
    parser = argparse.ArgumentParser(description='Cube fit example')
    parser.add_argument('--outdir', help='specify output directory', default='smpl_out')
    parser.add_argument('--discontinuous', action='store_true', default=False)
    parser.add_argument('--resolution', type=int, default=32, required=False)
    parser.add_argument('--mp4save', action='store_true', default=False)
    parser.add_argument('--n_rand_poses', type=int, default=2, required=False)
    parser.add_argument('--n_interp_pose_size', type=int, default=2, required=False)
    parser.add_argument('--n_camera_poses', type=int, default=100, required=False)
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
        mp4save_fn='progress.mp4',
        fit_mode='tex_mlp'
    )
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()