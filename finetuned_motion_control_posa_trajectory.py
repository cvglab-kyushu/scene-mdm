# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import sys
sys.path.append("/diffusion")
sys.path.append("/")

from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from diffusion.respace import SpacedDiffusion
from model.model_blending import ModelBlender
from utils.fixseed import fixseed
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.parser_util import edit_inpainting_args
from utils.model_util import load_model_blending_and_diffusion
from utils.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion
from utils import dist_util
from model.cfg_sampler import wrap_model
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric, convert_humanml_to_266, recover_from_ric_from_266
from data_loaders.humanml_utils import get_inpainting_mask
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion, plot_3d_pose, plot_3d_motion_w_traj
import shutil
from scipy.ndimage import gaussian_filter1d

from data_loaders.humanml.common.quaternion import *    # for debug


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def interpolate_keyframe(motion_data, keyframe):
    batch_size, num_joints, num_xyz, num_frames = motion_data.shape
    if keyframe == 0:
        motion_data[:, :, :, keyframe] = motion_data[:, :, :, 1]
    elif keyframe == num_frames - 1:
        motion_data[:, :, :, keyframe] = motion_data[:, :, :, num_frames - 2]
    else:
        prev_frame = motion_data[:, :, :, keyframe - 1]  # 前のフレーム
        next_frame = motion_data[:, :, :, keyframe + 1]  # 次のフレーム
        motion_data[:, :, :, keyframe] = (prev_frame + next_frame) / 2  # 線形補間
    return motion_data


def main():
    args_list = edit_inpainting_args()
    args = args_list[0]
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml', 'humanml_266', 'humanml_mask', 'humanml_smplx'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              load_mode='train',
                              size=args.num_samples)  # in train mode, you get both text and motion.
    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion if args_list[0].filter_noise else SpacedDiffusion
    model, diffusion = load_model_blending_and_diffusion(args_list, data, dist_util.dev(), DiffusionClass=DiffusionClass)

    iterator = iter(data)
    input_motions, model_kwargs = next(iterator)
    input_motions = input_motions.to(dist_util.dev())


    # add inpainting mask according to args
    assert max_frames == input_motions.shape[-1]
    gt_frames_per_sample = {}


    # Load posa output data
    posa_output_path = args.posa_output_path
    posa_data = np.load(posa_output_path, allow_pickle=True).item()
    posa_pose = torch.Tensor(posa_data['data'])
    posa_pose = convert_humanml_to_266(posa_pose)
    posa_pose = data.dataset.t2m_dataset.transform(posa_pose).float()
    keyframe = posa_data['keyframes']
    motion_len = posa_pose.shape[0]
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
    n_joints = 22


    if args.text_condition != '':
        texts = [args.text_condition] * args.num_samples
        model_kwargs['y']['text'] = texts
    elif args.posa_output_path != '':
        texts = [posa_data['text']] * args.num_samples
        model_kwargs['y']['text'] = texts

    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'result'.format(name, args.seed, args.guidance_param))
        out_path += '_' + texts[0].replace(' ', '_').replace('.', '')[:50] + '_gp={:.2f}'.format(args.guidance_param)


    mask, keyframes = get_inpainting_mask(args.inpainting_mask, 
                                         input_motions.shape, 
                                         model_kwargs['y']['lengths'], 
                                         only_text=args.only_text_condition,
                                         keyframes=keyframe,
                                        #  keyframes=None
                                         )
    keyframes = np.tile(keyframe, len(mask))
    key_mask_vel = torch.zeros(model_kwargs['y']['mask'].shape[:-1] + (model_kwargs['y']['mask'].shape[-1]-1,), dtype=torch.bool).to(dist_util.dev())
    key_mask = torch.zeros(model_kwargs['y']['mask'].shape, dtype=torch.bool).to(dist_util.dev())
    for idx, frame in enumerate(keyframes):
        key_mask_vel[idx, 0, 0, frame - 1] = True
        key_mask_vel[idx, 0, 0, frame] = True
        key_mask[idx, 0, 0, frame] = True

    model_kwargs['y']['key_mask_vel'] = torch.tensor(key_mask_vel).to(dist_util.dev())
    model_kwargs['y']['key_mask'] = torch.tensor(key_mask).to(dist_util.dev())
    model_kwargs['y']['inpainting_mask'] = torch.tensor(mask).float().to(dist_util.dev())


    # # 入力モーションをkeypose以外平均値に
    mean_motion = torch.Tensor(data.dataset.mean).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(args.num_samples, 1, 196, 1).permute(0, 3, 1, 2).to(args.device)
    mean_motion *= (1. - model_kwargs['y']['inpainting_mask'])
    input_motions = input_motions * model_kwargs['y']['inpainting_mask'] + mean_motion


    # # # key位置のx平面に対してy座標を反転
    # init = posa_pose[keyframe[0], 4].clone()
    # posa_pose[:, 4] = 2 * init - posa_pose[:, 4]
    # # y軸周りの回転行列（90度 = π/2）
    # theta = np.pi / 2
    # rotation_matrix = np.array([
    #     [np.cos(theta), 0, np.sin(theta)],
    #     [0, 1, 0],
    #     [-np.sin(theta), 0, np.cos(theta)]
    # ])

    # # 軌道全体を回転
    # rotated_trajectory = posa_pose.clone()[:, 4:7]

    # # y軸基準で平行移動（基準点を原点に）
    # rotated_trajectory[:, [0, 2]] -= posa_pose[keyframe[0], [4, 6]]

    # # 回転行列を適用
    # rotated_trajectory = rotated_trajectory @ rotation_matrix.T

    # # y軸基準の位置に戻す
    # rotated_trajectory[:, [0, 2]] += posa_pose[keyframe[0], [4, 6]]

    # posa_pose[:, 4:7] = rotated_trajectory

    # モーションのGlobal座標をPath Planningにより求めた座標に反映
    for i in range(input_motions.shape[3]):
        if i < len(posa_pose):
            input_motions[:,4,:,i] = posa_pose[i][4]
            input_motions[:,6,:,i] = posa_pose[i][6]
        else:
            input_motions[:,4,:,i] = posa_pose[-1][4]
            input_motions[:,6,:,i] = posa_pose[-1][6]

    if not args.only_text_condition:
        # KeyPoseの差し込み
        for i in range(len(keyframes)):
            # k = keyframes[i][1]
            k = keyframes[i]
            # input_motions[i][:, 0, k][5] = torch.Tensor(posa_pose[k][5])
            # input_motions[i][:, 0, k][7:] = torch.Tensor(posa_pose[k][7:])
            height = input_motions[0, 5, 0, 0]
            input_motions[i][:, 0, k] = torch.Tensor(posa_pose[k])
            input_motions[i][:, 0, k] = torch.Tensor(posa_pose[k])
            input_motions[i][5, 0, k] = height


            # # 角度反映
            # inp = input_motions.permute(0, 2, 3, 1)
            # q = inp[..., :4]
            # q_key = torch.Tensor(posa_pose[k, :4]).expand(q.shape).to(args.device)
            # new_q = qmul(q, q_key)
            # inp[..., :4] = new_q
            # input_motions = inp.permute(0, 3, 1, 2)


            # # 複数のキーポーズを挿入
            # keyf = keyframes[i]
            # for k in keyf:
            #     input_motions[i][:, 0, k][5] = torch.Tensor(posa_pose[0][5])
            #     input_motions[i][:, 0, k][7:] = torch.Tensor(posa_pose[0][7:])

            # # # 180度回転
            # quat = input_motions[i][:, 0, k][:4]
            # R = quaternion_to_matrix(quat).to(dist_util.dev())
            # quat_rot = matrix_to_quaternion(torch.matmul(rotmat, R)).to(dist_util.dev())
            # input_motions[i][:, 0, k][:4] = quat_rot 

    
    # # 20フレームずつズラしてkeyposeを挟み込んだ入力モーションとマスクを作成
    # input_motions = torch.zeros([10, 263, 1, 196])
    # input_masks = torch.zeros([10, 263, 1, 196])
    # for i in range(len(input_motions)):
    #     inmotion = torch.zeros([196, 263])
    #     mask = torch.zeros([196, 263])
    #     for frame in range(len(inmotion)):
    #         mask[frame][:3] = 1
    #         inmotion[frame] = torch.Tensor(mean)
    #         if frame == i * 20:
    #             mask[frame] = 1
    #             inmotion[frame] = torch.Tensor(posa_pose[0])
    #     input_motions[i] = inmotion.T.unsqueeze(1)
    #     input_masks[i] = mask.T.unsqueeze(1)
    # input_motions = input_motions.to(dist_util.dev())
    # model_kwargs['y']['inpainted_motion'] = input_motions
    # model_kwargs['y']['inpainting_mask'] = input_masks.float().to(dist_util.dev())

    # 回転軸と角度を定義
    # axis = torch.tensor([0.0, 1.0, 0.0])  # y軸周りに回転
    # angle = torch.tensor(180.0)  # 180度回転
    # angle_rad = torch.deg2rad(angle / 2)
    # rotation_quat = euler2quat(torch.tensor([0, torch.pi, 0]), order='xyz').to(dist_util.dev())

    # for debug


    
    # input_motions = input_motions.cpu().permute(0, 2, 3, 1)
    # input_motions = data.dataset.t2m_dataset.inv_transform(input_motions).float()

    # r = euler2quat(torch.tensor([[0, 90, 0]]), order="xyz", deg=True)
    # # inp = input_motions.permute(0, 2, 3, 1)
    # inp = input_motions
    # q = inp[..., :4]
    # r = r.expand(q.shape)
    # new_q = qmul(q, r)
    # inp[..., :4] = new_q
    # # input_motions = inp.permute(0, 3, 1, 2)

    # input_motions = recover_from_ric_from_266(input_motions, n_joints)
    # input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
    # motion = input_motions[0].transpose(2, 0, 1)
    # animation_save_path = os.path.join("./save/tmp/input_motion_0_key.mp4")
    # plot_3d_motion(animation_save_path, skeleton, motion, title="",
    #             dataset=args.dataset, fps=fps, vis_mode='gt')



    all_motions = []
    all_hml_motions = []
    all_text = []
    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')


        if args.dataset == 'humanml_mask':
            pad = torch.zeros([model_kwargs['y']['inpainting_mask'].shape[0], 1, 1, model_kwargs['y']['inpainting_mask'].shape[-1]])
            model_kwargs['y']['inpainting_mask'] = torch.cat([model_kwargs['y']['inpainting_mask'], pad.to(dist_util.dev())], dim=1)
            input_motions = torch.cat([input_motions, model_kwargs['y']['key_mask']], axis=1)

        input_motions = input_motions.to(dist_util.dev())
        model_kwargs['y']['inpainted_motion'] = input_motions

        # add CFG scale to batch
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=input_motions,
            progress=True,
            dump_steps=None,
            noise=None, 
            const_noise=False,
        )


        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22
            all_hml_motions.append(sample.cpu().numpy()[:, :, 0, ])
            sample = sample.cpu().permute(0, 2, 3, 1)
            if args.dataset == 'humanml_mask':
                sample = sample[:,...,:-1]
            sample = data.dataset.t2m_dataset.inv_transform(sample).float()
            sample = recover_from_ric_from_266(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        all_text += model_kwargs['y']['text']
        all_motions.append(sample.cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)

    # interpolate keyframe
    all_motions = interpolate_keyframe(all_motions, keyframe[0])


    # v = torch.tensor(all_motions[...,1:] - all_motions[...,:-1]).to(dist_util.dev())
    # a = model_kwargs['y']['key_mask'] * v
    # print("Key Pose Velocity Error: ", torch.sum(torch.abs(a)))
    

    all_hml_motions = np.concatenate(all_hml_motions, axis=0).transpose(0, 2, 1)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]

    # if os.path.exists(out_path):
    #     shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    t_fm_orig = np.tile(posa_data['t_fm_orig'], (len(all_motions), 1, 1))
    R_fm_orig = np.tile(posa_data['R_fm_orig'], (len(all_motions), 1, 1))
    # keyframes = [None] * len(all_motions)
    all_lengths = np.tile(motion_len, len(all_motions))
    np.save(npy_path,
            {'motion': all_motions, 'hml_motion': all_hml_motions, 'text': all_text, 'lengths': all_lengths, 
             'keyframes': keyframes, 't_fm_orig': t_fm_orig, 'R_fm_orig': R_fm_orig, 
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    # print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        input_motions = input_motions.cpu().permute(0, 2, 3, 1)
        if args.dataset == 'humanml_mask':
            input_motions = input_motions[:,...,:-1]
        input_motions = data.dataset.t2m_dataset.inv_transform(input_motions).float()
        # input_motions = recover_from_ric(input_motions, n_joints)
        input_motions = recover_from_ric_from_266(input_motions, n_joints)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()


    for sample_i in range(args.num_samples):
        rep_files = []
        if args.show_input:
            caption = 'Input Motion'
            # length = max_frames if args.only_text_condition else model_kwargs['y']['lengths'][sample_i]
            motion = input_motions[sample_i].transpose(2, 0, 1)[:motion_len]
            # motion = gaussian_filter1d(motion, 1, axis=0)
            save_file = 'input_motion{:02d}.mp4'.format(sample_i)
            if os.path.exists(save_file):
                continue
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | -> {save_file}]')
            # plot_3d_motion_w_traj(animation_save_path, skeleton, motion, title="",
            #             dataset=args.dataset, fps=fps, vis_mode='gt',
            #             gt_frames=gt_frames_per_sample.get(sample_i, []), kframes=kframes)
            plot_3d_motion(animation_save_path, skeleton, motion, title="",
                        dataset=args.dataset, fps=fps, vis_mode='gt',
                        gt_frames=gt_frames_per_sample.get(sample_i, []))
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            if args.guidance_param == 0:
                caption = 'Edit [{}] unconditioned'.format(args.inpainting_mask)
            else:
                caption = 'Edit [{}]: {}'.format(args.inpainting_mask, caption)
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:motion_len]


            # # Save joint speed graph
            # hml_motion = all_hml_motions[rep_i*args.batch_size + sample_i][:motion_len][:,7:]
            # save_speed = out_path + "/joint_speed/{:02d}_speed".format(rep_i*args.batch_size + sample_i)
            # os.makedirs(os.path.dirname(save_speed), exist_ok=True)
            # hml_motion = all_hml_motions[rep_i*args.batch_size + sample_i][:motion_len][:,7:]
            # speed = np.mean(np.abs(hml_motion[1:] - hml_motion[:-1]), axis=1)
            # kf = keyframes[rep_i*args.batch_size + sample_i]
            # plt.figure(figsize=(10, 6))
            # plt.plot(range(len(speed)), speed, label='Average Speed')
            # plt.axvline(x=kf, color='r', linestyle='--', label='Keyframe (Frame {})'.format(kf))
            # plt.xlabel('Frame')
            # plt.ylabel('Speed')
            # plt.title('Transition of Speed')
            # plt.legend()
            # plt.grid(True)
            # plt.savefig(save_speed)

            # # Save root angle speed graph
            # hml_motion = all_hml_motions[rep_i*args.batch_size + sample_i][:motion_len][:,3:7]
            # save_speed = out_path + "/joint_root_angle/{:02d}_speed".format(rep_i*args.batch_size + sample_i)
            # os.makedirs(os.path.dirname(save_speed), exist_ok=True)
            # hml_motion = all_hml_motions[rep_i*args.batch_size + sample_i][:motion_len][:,7:]
            # speed = np.mean(np.abs(hml_motion[1:] - hml_motion[:-1]), axis=1)
            # kf = keyframes[rep_i*args.batch_size + sample_i]
            # plt.figure(figsize=(10, 6))
            # plt.plot(range(len(speed)), speed, label='Average Speed')
            # plt.axvline(x=kf, color='r', linestyle='--', label='Keyframe (Frame {})'.format(kf))
            # plt.xlabel('Frame')
            # plt.ylabel('Speed')
            # plt.title('Transition of Speed')
            # plt.legend()
            # plt.grid(True)
            # plt.savefig(save_speed)



            # motion = gaussian_filter1d(motion, 1, axis=0)
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {animation_save_path}]')
            # plot_3d_motion_w_traj(animation_save_path, skeleton, motion, title="",
            #                dataset=args.dataset, fps=fps, vis_mode=args.inpainting_mask,
            #                gt_frames=gt_frames_per_sample.get(sample_i, []), kframes=kframes)
            plot_3d_motion(animation_save_path, skeleton, motion, title="",
                            dataset=args.dataset, fps=fps, vis_mode=args.inpainting_mask,
                            gt_frames=gt_frames_per_sample.get(sample_i, []))
                        # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        if args.num_repetitions > 1:
            all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
            ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
            hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions + (1 if args.show_input else 0)} '
            ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
            os.system(ffmpeg_rep_cmd)
            print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()