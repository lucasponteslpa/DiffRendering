import argparse
import os
import pathlib
import sys
import numpy as np
import torch
import imageio
import numpy as np
import cv2
import pywavefront

from samples.torch import util

import nvdiffrast.torch as dr
from texture_tool.models.smpl_torch import SMPLModel

from render.obj import load_obj
from render.render import interpolate
import utils

from texture_tool.models.smpl import Smpl

def make_grid(arr, ncols=2):
    n, height, width, nc = arr.shape
    nrows = n//ncols
    assert n == nrows*ncols
    return arr.reshape(nrows, ncols, height, width, nc).swapaxes(1,2).reshape(height*nrows, width*ncols, nc)

def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def simple_render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, resolution: int):
    pos_clip    = transform_pos(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    color, _    = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color

def texture_render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip=True, max_mip_level=9, mip=None):
    # breakpoint()
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    # res = [944,1288]
    # rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=res)
    if enable_mip:
        texc, texd = dr.interpolate(uv, rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex, texc, texd, mip=mip ,filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv, rast_out, uv_idx)
        color = dr.texture(tex, texc, filter_mode='linear')
    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color

def make_grid(arr, ncols=2):
    n, height, width, nc = arr.shape
    nrows = n//ncols
    assert n == nrows*ncols
    return arr.reshape(nrows, ncols, height, width, nc).swapaxes(1,2).reshape(height*nrows, width*ncols, nc)

def gen_smpl_torch_model(model_path='texture_tool/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl', gpu_id=[0]):
    if len(gpu_id) > 0 and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id[0])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = SMPLModel(device=device, model_path=model_path)
    return model, device

def gen_rand_smpl_param(device,pose_size=72, beta_size=10, n_base_points=5, interp_size=100):
    base_idx = np.array(list(range(n_base_points)))
    interp_array = np.array([(i)/interp_size for i in range(interp_size*(n_base_points-1))])

    pose = np.array([(np.interp(interp_array, base_idx,np.random.rand( n_base_points)) - 0.5) for _ in range(pose_size)])
    pose = torch.from_numpy(pose).type(torch.float64).to(device)
    betas = np.array([(np.interp(interp_array, base_idx,np.random.rand( n_base_points)) - 0.5)*0.06 for _ in range(beta_size)])
    betas = torch.from_numpy(betas).type(torch.float64).to(device)
    trans = torch.from_numpy(np.zeros((3,interp_size*n_base_points))).type(torch.float64).to(device)

    return pose.transpose(1,0), betas.transpose(1,0), trans.transpose(1,0)

def load_rand_smpl_param(device,path_prefix='NeuralActor_dataset/lan/training/transform_smoth3e-2_withmotion/', n_frames=500):
    poses, trans = [], []
    rodrigues = lambda x : cv2.Rodrigues(x)[0]
    for i in range(n_frames):
        str_index = str(i)
        param_path = os.path.join(path_prefix,(6-len(str_index))*'0'+str_index+'.json')
        joints, translation, rotation, pose = utils.get_SMPL_parametes_from_json(param_path)
        Rh = rodrigues(rotation.transpose(1,0))[:,0]
        shape_data_pose = np.concatenate([Rh[None, :], pose[:, 3:]], -1)
        poses.append(shape_data_pose)
        trans.append(translation)
    poses = torch.from_numpy(np.array(poses)).type(torch.float64).to(device)
    trans = torch.from_numpy(np.array(trans)).type(torch.float64).to(device)

    return poses, trans

def gen_smpl_vertices(model_path='texture_tool/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl', gpu_id=[0]):
    if len(gpu_id) > 0 and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id[0])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)

    pose_size = 72
    beta_size = 10

    # np.random.seed(9608)
    pose = torch.from_numpy((np.random.rand(pose_size) - 0.5))\
            .type(torch.float64).to(device)
    betas = torch.from_numpy((np.random.rand(beta_size) - 0.5) * 0.06) \
            .type(torch.float64).to(device)
    trans = torch.from_numpy(np.zeros(3)).type(torch.float64).to(device)
    # outmesh_path = './smpl_torch.obj'

    model = SMPLModel(device=device, model_path=model_path)
    result = model(betas, pose, trans)
    return result.float(), model, device

def gen_smpl_rand_1_cam_mp4(ctx, writer, mesh, smpl_model, poses, betas, trans, mtx, resolution, n_cam=4):
    for p, b, t in zip(poses,betas, trans):
        img_frames = []
        for m in mtx[:n_cam] if n_cam < len(mtx) else mtx:
            f = gen_img_frame(ctx,mesh, smpl_model, p, b, t, m,resolution)
            f = np.clip(np.rint(f[0].cpu().detach().numpy()*255),0,255).astype(np.uint8)
            img_frames.append(f)
        img_grid = make_grid(np.stack(img_frames))
        writer.append_data(img_grid)
    writer.close()

def gen_smpl_rand_1_cam_mp4_rbetas(ctx, writer, mesh, smpl_model, poses, trans, mtx, resolution):
    b = torch.from_numpy((np.random.rand(10) - 0.5) * 0.06) \
            .type(torch.float64).to('cuda')
    count = 0
    for p, t in zip(poses, trans):
        img_frames = []
        for m in mtx:
            f = gen_img_frame(ctx,mesh, smpl_model, p, b, t, m,resolution)
            f = np.clip(np.rint(f[0].cpu().detach().numpy()*255),0,255).astype(np.uint8)
            if count==0:
                cv2.imwrite('out_frame.png',f[:,:,::-1])
                count += 1
            writer.append_data(f)
            # img_frames.append(f)
        # img_grid = make_grid(np.stack(img_frames))
        # img_grid = make_grid(f)
    writer.close()

def simple_smpl_render(ctx, mesh, mtx_in, resolution, enable_mip=True):
    color = texture_render(ctx,
                           mtx_in,
                           mesh.v_pos,
                           mesh.t_pos_idx.to(torch.int32),
                           mesh.v_tex,
                           mesh.t_tex_idx.to(torch.int32),
                           mesh.material['kd'].data,
                           resolution,
                           enable_mip=enable_mip)
    return color

def gen_img_frame(ctx, mesh, smpl_model, pose, betas, trans, mtx, resolution, verts=None):
    if verts is None:
        result_vert = smpl_model(betas, pose, trans)
    else:
        result_vert = verts
    mesh.v_pos = result_vert.type(torch.float32)
    color = simple_smpl_render(ctx, mesh, mtx, resolution)
    return color

def main():
    parser = argparse.ArgumentParser(description='Cube fit example')
    parser.add_argument('--outdir', help='specify output directory', default='smpl_out')
    parser.add_argument('--discontinuous', action='store_true', default=False)
    parser.add_argument('--resolution', type=int, default=32, required=False)
    parser.add_argument('--mp4save', action='store_true', default=False)
    args = parser.parse_args()

    writer = imageio.get_writer(f'{args.outdir}/smpl_rand_pose_1_cam.mp4', mode='I', fps=50, codec='libx264', bitrate='16M')

    smpl = Smpl(model='texture_tool/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    # smpl_f_mesh = load_obj('texture_tool/smpl_model/smpl_sample/SMPL/SMPL_female_default_resolution.obj',
    #                         mtl_override='texture_tool/smpl_model/smpl_sample/SMPL/SMPL_female_default_resolution.mtl')
    smpl_f_mesh = load_obj('NeuralActor_dataset/lan/uvmapping.obj', mtl_override='NeuralActor_dataset/lan/lan.mtl')
    glctx = dr.RasterizeCudaContext()

    smpl_vert, smpl_model, device = gen_smpl_vertices(gpu_id=[1])
    poses, betas, trans = gen_rand_smpl_param(device)
    # Random rotation/translation matrix for optimization.
    mtx_list = []
    # for i in range(4):

    #     r_rot = util.random_rotation_translation(0.25 + i*0.05)

    #     # Modelview and modelview + projection matrices.
    #     proj  = util.projection(x=0.4)
    #     r_mv  = np.matmul(util.translate(0, 0, -3.5), r_rot)
    #     r_mvp = np.matmul(proj, r_mv).astype(np.float32)
    #     mtx_list.append(r_mvp)
    # Smooth rotation for display.
    # ang=0
    # a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))
    # a_mv  = np.matmul(util.translate(0, 0, -3.5), a_rot)
    # a_mvp = np.matmul(proj, a_mv).astype(np.float32)

    # gen_smpl_rand_1_cam_mp4(glctx,writer,smpl_f_mesh, smpl_model, poses, betas, trans, mtx_list, args.resolution)
    print("Done.")
    rodrigues = lambda x : cv2.Rodrigues(x)[0]

    intrinsic = utils.load_intrinsics('NeuralActor_dataset/lan/intrinsic/0_train_0010.txt')
    extrinsic = utils.load_matrix('NeuralActor_dataset/lan/pose/0_train_0010.txt')
    extrinsic = utils.parse_extrinsics(extrinsic,False)
    # extrinsic = np.linalg.inv(extrinsic)
    # breakpoint()


    # r_mvp = pose@intrinsic.T
    r_rot = util.random_rotation_translation(0.25)
    l_rot = utils.look_at_rotation(torch.from_numpy(extrinsic[:-1,-1]))
    # breakpoint()
    # extrinsic[:-1,:-1] = l_rot.numpy()
    # r_rot = util.random_rotation_translation(0.25, r_mvp[:3,:3])
    proj  = util.projection(x=2.0,f=50.0)
    # proj  = util.projection_asp(r=1.36,t=1.0,f=50.0)
    r_mv  = np.matmul(util.translate(0,0,-4.5), r_rot)
    mv  = np.matmul(util.translate(0,0,-5.55), extrinsic)
    I = np.eye(4)
    I[0 ,0] = -I[0,0]
    mv_eye = np.matmul(util.translate(0,0,-5.55), np.matmul(extrinsic,I))
    # mv_eye = np.matmul(extrinsic,I)
    # r_mvp = np.matmul(proj, pose*0.01).astype(np.float32)
    r_mvp = np.matmul(proj, r_mv).astype(np.float32)
    mvp = np.matmul(proj, mv).astype(np.float32)
    mvp_eye = np.matmul(proj, mv_eye).astype(np.float32)
    poses, trans = load_rand_smpl_param(device,n_frames=1000)


    c_mtx = np.array([[ -1,   1,  1, 1],
                      [ -1,   1,  1, 1],
                      [  1,  -1, -1, 1],
                      [  1,   1,  1, 1]]).astype(np.float32)
    # for i,a in enumerate(np.arange(-np.pi,np.pi,np.pi/1000)):
    for _ in range(1):
        I = np.eye(4)
        m = np.eye(4)
        I[0 ,0] = -I[0,0]
        # I[1 ,1] = -I[1,1]
        # I[2 ,2] = -I[2,2]
        # Modelview and modelview + projection matrices.
        # proj  = util.projection(x=0.78,f=50.0)
        proj  = util.projection_intrinsic(intrinsic)
        # proj  = util.projection_asp(r=1.36,t=1.0,f=50.0)
        # rot = np.matmul(util.rotate_x(a), I)
        rot = np.eye(4)
        rot[:-1,:-1] = extrinsic[:-1,:-1]
        # rot = np.matmul(rot, I)
        # # mv_ext = np.matmul(util.translate(*(-np.matmul(rot[:-1,:-1],extrinsic[:3,3]))), rot)
        # mv_ext = np.matmul(util.translate(*(extrinsic[:3,3])), rot)
        # mvp = np.matmul(proj, mv_ext).astype(np.float32)
        C = np.matmul(I[:-1,:-1],extrinsic[:3,3])
        # C = extrinsic[:3,3]
        L = - C
        L_n = L/np.linalg.norm(L)
        s = np.cross(L,np.array([0,-1,0]))
        s_n = s/np.linalg.norm(s)
        u = np.cross(s_n,L_n)
        R = np.stack([s_n,u,-L_n])
        rot_mtx = c_mtx*rot.transpose(1,0)
        # rot_mtx[0,1] = rot_mtx[0,1] - 0.02
        # R = np.matmul(util.rotate_z(a)[:-1,:-1],rot[:-1,:-1])

        R = rot_mtx[:-1,:-1]
        t = -np.matmul(R,C)
        m[:-1,:-1] = R
        m[:3,3] = t
        mv = np.matmul(m,I)
        # mv  = np.matmul(util.translate(0, 0, -1.5), mv)
        mvp = np.matmul(proj, mv).astype(np.float32)
        mtx_list.append(torch.from_numpy(mvp).to(device))

    smpl.pose = poses[0].cpu().detach().numpy()
    smpl._set_up()
    v_smpl = torch.from_numpy(np.array(smpl.v)).to(device)
    v_smpl = v_smpl + trans[0]
    f_smpl=gen_img_frame(glctx,
                  smpl_f_mesh,
                  smpl_model,
                #   torch.from_numpy(joints).to(device).view(-1)*0.2,
                  poses[0],
                  betas[0],
                  trans[0],
                  mtx_list[0],
                  args.resolution,
                  verts=v_smpl)

    breakpoint()
    cv2.imwrite('frame_smpl.png', np.clip(np.rint(f_smpl[0].cpu().detach().numpy()*255),0,255).astype(np.uint8)[:,:,::-1])
    gen_smpl_rand_1_cam_mp4_rbetas(glctx,writer, smpl_f_mesh, smpl_model, poses, trans, mtx_list, args.resolution)

    # joints, translation, rotation, pose = utils.get_SMPL_parametes_from_json('NeuralActor_dataset/oleks/training/transform_smoth3e-2_withmotion/000800.json')
    # Rh = rodrigues(rotation.transpose(1,0))[:,0]
    # shape_data_pose = np.concatenate([Rh[None, :], pose[:, 3:]], -1)
    # f=gen_img_frame(glctx,
    #               smpl_f_mesh,
    #               smpl_model,
    #             #   torch.from_numpy(joints).to(device).view(-1)*0.2,
    #               torch.from_numpy(shape_data_pose).to(device).view(-1),
    #               betas[0],
    #               torch.from_numpy(translation).to(device),
    #               mvp,
    #               args.resolution)
    # cv2.imwrite('frame.png', np.clip(np.rint(f[0].cpu().detach().numpy()*255),0,255).astype(np.uint8)[:,:,::-1])
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()