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

from mlp_models.mlp_models import SimpleColorMLP, NeuralTextureSMPL

from render.texture import texture2d_mip

from smpl_visualize import *

from neural_actor_dataset import NeuralActorDataset

def model_inf_tex(dataset,model,path, index=0, posfix=''):
    frame_data = dataset[index]
    tex_inf = model(frame_data['pose'].to(torch.float32)    )
    cv2.imwrite(f'{path}/tex_inf{index}_{posfix}.png',tex_inf.cpu().detach().numpy()[0,:,:,::-1]*255)

def generate_video(gctx, writer, dataset, model, smpl_model,smpl_mesh,tex_shape,resolution,device='cuda', camera=0):
    b = torch.from_numpy((np.zeros(10))).type(torch.float64).to(device)
    for frame_data in dataset:
        smpl_v = smpl_model(b, frame_data['pose'], frame_data['translation'])
        smpl_mesh.v_pos = smpl_v.type(torch.float32)
        dataset.set_camera_idx(camera)
        img_frames = []
        for _ in range(2):
            frame_dict = dataset.get_frame()
            mvp = frame_dict['mvp']
            out_tex = model(frame_data['pose'].to(torch.float32))
            mips = [out_tex]
            while mips[-1].shape[1] > 1 and mips[-1].shape[2] > 1:
                mips += [texture2d_mip.apply(mips[-1])]
            f_inf = texture_render(gctx,
                            mvp,
                            smpl_mesh.v_pos,
                            smpl_mesh.t_pos_idx.to(torch.int32),
                            smpl_mesh.v_tex,
                            smpl_mesh.t_tex_idx.to(torch.int32),
                            out_tex,
                            resolution,
                            mip=mips[1:])
            f = np.clip(frame_dict['target_frame'].cpu().detach().numpy()*255,0,255).astype(np.uint8)
            img_frames.append(f)
            f = np.clip(f_inf[0].cpu().detach().numpy()*255,0,255).astype(np.uint8)
            img_frames.append(f)

            dataset.next_camera()
        img_grid = make_grid(np.stack(img_frames))
        writer.append_data(img_grid)

    writer.close()



def fit_smpl(gctx,
             dataset,
             smpl_model,
             smpl_mesh,
             tex_shape,
             resolution,
             device            = 'cuda',
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

    model = NeuralTextureSMPL(tex_shape).to(device)
    b = torch.from_numpy((np.zeros(10))).type(torch.float64).to(device)

    # Adam optimizer for vertex position and color with a learning rate ramp.
    optimizer    = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.01, 10**(-x*0.0005)))
    ssim = SSIM(data_range=1.0)
    model_inf_tex(dataset,model,'smpl_out',posfix='initial')
    with tqdm(total=len(dataset)+1) as tq:
        for it in range(max_iter):
            print("Epoch {}/{}".format(it+1, max_iter))
            save_mp4 = mp4save_interval and (it % mp4save_interval == 0)

            for frame_data in dataset:
                smpl_v = smpl_model(b, frame_data['pose'], frame_data['translation'])
                smpl_mesh.v_pos = smpl_v.type(torch.float32)
                dataset.reset_camera()
                total_loss = 0
                for _ in range(dataset.get_camera_len()):
                # for _ in range(1):
                    frame_dict = dataset.get_frame()
                    mvp = frame_dict['mvp']
                    out_tex = model(frame_data['pose'].to(torch.float32))
                    mips = [out_tex]
                    while mips[-1].shape[1] > 1 and mips[-1].shape[2] > 1:
                        mips += [texture2d_mip.apply(mips[-1])]
                    color_opt = texture_render(gctx,
                                    mvp,
                                    smpl_mesh.v_pos,
                                    smpl_mesh.t_pos_idx.to(torch.int32),
                                    smpl_mesh.v_tex,
                                    smpl_mesh.t_tex_idx.to(torch.int32),
                                    out_tex,
                                    resolution,
                                    mip=mips[1:])
                    loss = torch.mean((frame_dict['target_frame'].unsqueeze(0) - color_opt)**2) + (1.0 - ssim(frame_dict['target_frame'].unsqueeze(0).permute(0,3,1,2),color_opt.permute(0,3,1,2)))/2.0
                    total_loss += loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                tq.set_postfix({'loss':loss.item()})
                tq.update()
            print("Mean Epoch {} Loss: {}".format(it+1, total_loss/len(dataset)))

            tq.refresh()
            tq.reset()
    model_inf_tex(dataset,model,'smpl_out',posfix='final')
    # Done.
    if writer is not None:
        generate_video(gctx, writer,dataset,model,smpl_model,smpl_mesh, tex_shape,512,device)



def main():
    parser = argparse.ArgumentParser(description='Cube fit example')
    parser.add_argument('--outdir', help='specify output directory', default='smpl_out')
    parser.add_argument('--discontinuous', action='store_true', default=False)
    parser.add_argument('--resolution', type=int, default=1024, required=False)
    parser.add_argument('--mp4save', action='store_true', default=False)
    parser.add_argument('--n_rand_poses', type=int, default=5, required=False)
    parser.add_argument('--n_interp_pose_size', type=int, default=100, required=False)
    parser.add_argument('--n_camera_poses', type=int, default=10, required=False)
    parser.add_argument('--display-interval', type=int, default=0)
    parser.add_argument('--mp4save-interval', type=int, default=100)
    parser.add_argument('--max-iter', type=int, default=5)
    args = parser.parse_args()

    dataset = NeuralActorDataset('data/')
    writer = imageio.get_writer(f'{args.outdir}/smpl_rand_pose_1_cam.mp4', mode='I', fps=20, codec='libx264', bitrate='16M')

    smpl_mesh = load_obj('texture_tool/smpl_model/smpl_sample/SMPL/SMPL_female_default_resolution.obj',mtl_override='texture_tool/smpl_model/smpl_sample/SMPL/SMPL_female_default_resolution.mtl')
    glctx = dr.RasterizeCudaContext()

    _, smpl_model, device = gen_smpl_vertices(gpu_id=[1])


    tex_shape = (1,512,512,3)
    fit_smpl(
        glctx,
        dataset,
        smpl_model,
        smpl_mesh,
        tex_shape,
        args.resolution,
        max_iter=args.max_iter,
        log_interval=10,
        display_interval=args.display_interval,
        out_dir=args.outdir,
        log_fn='log.txt',
        mp4save_interval=args.mp4save_interval,
        mp4save_fn='progress.mp4',
        fit_mode='tex'
    )
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()