import sys, os
import copy
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import codecs as cs
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin
import smplx

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from common.skeleton import Skeleton
from common.quaternion import *
from paramUtil import *
from plot_script import plot_3d_pose

POSA_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(POSA_dir), "visualize/joints2smpl/src"))
import config

# os.environ['PYOPENGL_PLATFORM'] = 'egl'

smplmodel = smplx.create(config.SMPL_MODEL_DIR, 
                         model_type="smplx", gender="neutral", ext="npz",
                         batch_size=1)

def convert_smplx2humanml(device, data, save_path, flip=True):

    ###########################################################################
    ################### raw_pose_processing.py in HumanML3D ################### 
    ###########################################################################


    trans_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, 1.0, 0.0]])
    body_model = smplmodel.to(device)

    def amass_to_pose(bdata):
        # if bdata['gender'] == 'male':
        #     bm = male_bm
        # else:
        #     bm = female_bm

        with torch.no_grad():
            root_orient = torch.Tensor(bdata['global_orient']).to(device) # controls the global root orientation
            pose_body = torch.Tensor(bdata['body_pose']).to(device) # controls the body
            pose_hand = None # controls the finger articulation
            betas = torch.Tensor(bdata['betas']).to(device) # controls the body shape
            trans = torch.Tensor(bdata['transl']).to(device) 
            pose_seq = []
            for i in range(len(root_orient)):  
                body = body_model(betas=betas[i][None], 
                                  global_orient=root_orient[i][None], 
                                  body_pose=pose_body[i][None], 
                                  transl=trans[i][None], 
                                  return_verts=True)
                pose_seq.append(body.joints[0, :22].detach().cpu().numpy())
        pose_seq_n = np.dot(np.array(pose_seq), trans_matrix)

        # # for debug
        # import trimesh
        # v = body.v.detach().cpu().numpy()
        # v += trans.unsqueeze(1).detach().cpu().numpy()
        # for i in range(body.v.shape[0]):
        #     mesh = trimesh.Trimesh(vertices=v[i], faces=body.f.detach().cpu().numpy())
        #     mesh.export("./save/tmp_mesh/body_{:03d}.ply".format(i))
        # for i in range(body.v.shape[0]):
        #     vr = np.matmul(bdata['R_fm_orig'], v[i].T).T
        #     mesh = trimesh.Trimesh(vertices=vr, faces=body.f.detach().cpu().numpy())
        #     mesh.export("./save/tmp_mesh_r/body_{:03d}.ply".format(i))

        # np.save(save_path, pose_seq_np_n)
        return pose_seq_n, bdata['t_fm_orig'], bdata['R_fm_orig'], bdata['keyframes'], bdata['length']

    if type(data) == str:
        bdata = pickle.load(open(data_path, 'rb'))
    else:
        bdata = data

    data, trans, orient, keyframes, length = amass_to_pose(bdata)
    # np.matmul(rot_mat, vertices_np.transpose())

    if len(data) == 1:
        data = np.vstack([data, data])

    def swap_left_right(data):
        assert len(data.shape) == 3 and data.shape[-1] == 3
        data = data.copy()
        data[..., 0] *= -1
        right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
        left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
        left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
        right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
        tmp = data[:, right_chain]
        data[:, right_chain] = data[:, left_chain]
        data[:, left_chain] = tmp
        if data.shape[1] > 24:
            tmp = data[:, right_hand_chain]
            data[:, right_hand_chain] = data[:, left_hand_chain]
            data[:, left_hand_chain] = tmp
        return data

    # data_m = swap_left_right(data)

    print("Done raw_pose_processing.")

    ###########################################################################
    ################## motion_representation.py in HumanML3D ################## 
    ###########################################################################


    def uniform_skeleton(positions, target_offset):
        src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
        src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
        src_offset = src_offset.numpy()
        tgt_offset = target_offset.numpy()
        # print(src_offset)
        # print(tgt_offset)
        '''Calculate Scale Ratio as the ratio of legs'''
        src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
        tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

        scale_rt = tgt_leg_len / src_leg_len
        # print(scale_rt)
        src_root_pos = positions[:, 0]
        tgt_root_pos = src_root_pos * scale_rt

        '''Inverse Kinematics'''
        quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
        # print(quat_params.shape)

        '''Forward Kinematics'''
        src_skel.set_offset(target_offset)
        new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
        return new_joints



    def process_file(positions, feet_thre):
        # (seq_len, joints_num, 3)
        #     '''Down Sample'''
        #     positions = positions[::ds_num]

        '''Uniform Skeleton'''
        positions = uniform_skeleton(positions, tgt_offsets)

        '''Put on Floor'''
        floor_height = positions.min(axis=0).min(axis=0)[1]
        positions[:, :, 1] -= floor_height
        #     print(floor_height)

        #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

        '''XZ at origin'''
        root_pos_init = positions[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
        positions = positions - root_pose_init_xz

        # '''Move the first pose to origin '''
        # root_pos_init = positions[0]
        # positions = positions - root_pos_init[0]

        '''All initially face Z+'''
        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

        # forward (3,), rotate around y-axis
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        # forward (3,)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

        #     print(forward_init)

        target = np.array([[0, 0, 1]])
        root_quat_init = qbetween_np(forward_init, target)
        root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

        positions_b = positions.copy()

        positions = qrot_np(root_quat_init, positions)

        #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

        '''New ground truth positions'''
        global_positions = positions.copy()

        # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
        # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
        # plt.xlabel('x')
        # plt.ylabel('z')
        # plt.axis('equal')
        # plt.show()

        """ Get Foot Contacts """

        def foot_detect(positions, thres):
            velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

            feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            #     feet_l_h = positions[:-1,fid_l,1]
            #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
            feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

            feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            #     feet_r_h = positions[:-1,fid_r,1]
            #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
            feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
            return feet_l, feet_r
        #
        feet_l, feet_r = foot_detect(positions, feet_thre)
        # feet_l, feet_r = foot_detect(positions, 0.002)

        '''Quaternion and Cartesian representation'''
        r_rot = None

        def get_rifke(positions):
            '''Local pose'''
            positions[..., 0] -= positions[:, 0:1, 0]
            positions[..., 2] -= positions[:, 0:1, 2]
            '''All pose face Z+'''
            positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
            return positions

        def get_quaternion(positions):
            skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
            # (seq_len, joints_num, 4)
            quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

            '''Fix Quaternion Discontinuity'''
            quat_params = qfix(quat_params)
            # (seq_len, 4)
            r_rot = quat_params[:, 0].copy()
            #     print(r_rot[0])
            '''Root Linear Velocity'''
            # (seq_len - 1, 3)
            velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
            #     print(r_rot.shape, velocity.shape)
            velocity = qrot_np(r_rot[1:], velocity)
            '''Root Angular Velocity'''
            # (seq_len - 1, 4)
            r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
            quat_params[1:, 0] = r_velocity
            # (seq_len, joints_num, 4)
            return quat_params, r_velocity, velocity, r_rot

        def get_cont6d_params(positions):
            skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
            # (seq_len, joints_num, 4)
            quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

            '''Quaternion to continuous 6D'''
            cont_6d_params = quaternion_to_cont6d_np(quat_params)
            # (seq_len, 4)
            r_rot = quat_params[:, 0].copy()
            #     print(r_rot[0])
            '''Root Linear Velocity'''
            # (seq_len - 1, 3)
            velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
            #     print(r_rot.shape, velocity.shape)
            velocity = qrot_np(r_rot[1:], velocity)
            '''Root Angular Velocity'''
            # (seq_len - 1, 4)
            r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
            # (seq_len, joints_num, 4)
            return cont_6d_params, r_velocity, velocity, r_rot

        cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
        positions = get_rifke(positions)

        #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
        #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

        # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
        # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
        # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
        # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
        # plt.xlabel('x')
        # plt.ylabel('z')
        # plt.axis('equal')
        # plt.show()

        '''Root height'''
        root_y = positions[:, 0, 1:2]

        '''Root rotation and linear velocity'''
        # (seq_len-1, 1) rotation velocity along y-axis
        # (seq_len-1, 2) linear velovity on xz plane
        r_velocity = np.arcsin(r_velocity[:, 2:3])
        l_velocity = velocity[:, [0, 2]]
        #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
        root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

        '''Get Joint Rotation Representation'''
        # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
        rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

        '''Get Joint Rotation Invariant Position Represention'''
        # (seq_len, (joints_num-1)*3) local joint position
        ric_data = positions[:, 1:].reshape(len(positions), -1)

        '''Get Joint Velocity Representation'''
        # (seq_len-1, joints_num*3)
        local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                            global_positions[1:] - global_positions[:-1])
        local_vel = local_vel.reshape(len(local_vel), -1)

        data = root_data
        data = np.concatenate([data, ric_data[:-1]], axis=-1)
        data = np.concatenate([data, rot_data[:-1]], axis=-1)
        #     print(data.shape, local_vel.shape)
        data = np.concatenate([data, local_vel], axis=-1)
        data = np.concatenate([data, feet_l, feet_r], axis=-1)

        return data, global_positions, positions, l_velocity


    def recover_root_rot_pos(data):
        rot_vel = data[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        '''Get Y-axis rotation from rotation velocity'''
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang)

        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
        '''Add Y-axis rotation to root position'''
        r_pos = qrot(qinv(r_rot_quat), r_pos)

        r_pos = torch.cumsum(r_pos, dim=-2)

        r_pos[..., 1] = data[..., 3]
        return r_rot_quat, r_pos


    def recover_from_rot(data, joints_num, skeleton):
        r_rot_quat, r_pos = recover_root_rot_pos(data)

        r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

        start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
        end_indx = start_indx + (joints_num - 1) * 6
        cont6d_params = data[..., start_indx:end_indx]
        #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
        cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
        cont6d_params = cont6d_params.view(-1, joints_num, 6)

        positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

        return positions


    def recover_from_ric(data, joints_num):
        r_rot_quat, r_pos = recover_root_rot_pos(data)
        positions = data[..., 4:(joints_num - 1) * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 2] += r_pos[..., 2:3]

        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

        return positions
    
    def flip_human_pose(data, kinematic_chain):
        # Flip horizontally: Swap [0 and 1] and [3 and 4]
        flip = np.copy(data)
        for chain_0, chain_1 in zip(kinematic_chain[0], kinematic_chain[1]):
            flip[:, chain_0], flip[:, chain_1] = np.copy(flip[:, chain_1]), np.copy(flip[:, chain_0])
        for chain_3, chain_4 in zip(kinematic_chain[3], kinematic_chain[4]):
            flip[:, chain_3], flip[:, chain_4] = np.copy(flip[:, chain_4]), np.copy(flip[:, chain_3])
        return flip


    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 22
    # ds_num = 8

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    # Get offsets of target skeleton
    example_data = data.reshape(len(data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    # print(tgt_offsets)

    frame_num = 0
    source_data = data[:, :joints_num]

    ref = source_data[0,0,:]
    source_data -= ref
    source_data += trans

    if flip:
        # flip horizontally
        source_data = flip_human_pose(source_data, kinematic_chain)

    data, ground_positions, positions, l_velocity = process_file(source_data, 0.002)
    # rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)

    save_dict = copy.deepcopy(bdata)

    # # Insert key pose
    # if max(keyframes) < len(data):
    #     for i in range(len(keyframes)):
    #         kf = keyframes[i]
    #         data_x = copy.deepcopy(data[kf][4])
    #         data_z = copy.deepcopy(data[kf][6])
    #         data[kf] = bdata['key_pose'][i]
    #         data[kf][4] = data_x
    #         data[kf][6] = data_z

    save_dict['data'] = data

    np.save(save_path, save_dict)

if __name__ == "__main__":
    data_path = '../save/humanml_only_text_condition/result_a_person_walks_and_sits_on_a_chair/sample01_rep00_iter=20_new_opt_cam_t_affordance/pkl/MPH11/048.pkl'
    save_path = '../save/humanml_only_text_condition/result_a_person_walks_and_sits_on_a_chair/sample01_rep00_iter=20_new_opt_cam_t_affordance/pkl/MPH11/048.npy'

    male_bm_path = './body_models/smplh/male/model.npz'
    male_dmpl_path = './body_models/dmpls/male/model.npz'

    female_bm_path = './body_models/smplh/female/model.npz'
    female_dmpl_path = './body_models/dmpls/female/model.npz'

    num_betas = 10 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to(comp_device)
    male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to(device)

    convert_smplx2humanml(device, data_path, save_path)

print()