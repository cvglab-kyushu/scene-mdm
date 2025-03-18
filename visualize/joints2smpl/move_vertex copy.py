import os, sys
sys.path.append(os.getcwd())

import glob
import copy
import cv2
import joblib
import trimesh
import open3d as o3d
import numpy as np
import torch
import torchgeometry as tgm
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import transformations as tf

from human_body_prior.body_model.body_model import BodyModel
from POSA.src import posa_utils, eulerangles, viz_utils, misc_utils, data_utils, opt_utils


R_smpl2scene = eulerangles.euler2mat(np.pi / 2, 0, 0, 'sxyz')


def plot_vector1(aa, trans, v, save_path=None):
    R = tgm.angle_axis_to_rotation_matrix(torch.Tensor(aa).unsqueeze(0))[0, :3, :3]
    # R = np.dot(R_smpl2scene, R)
    vector = np.array([0, 0, 1])
    rotated_vector = np.dot(R, vector)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(trans[0], trans[1], trans[2], rotated_vector[0], rotated_vector[1], rotated_vector[2], color='r')
    ax.scatter(trans[0], trans[1], trans[2], color='r', marker='o', label='Point1')
    ax.scatter(v[:,0], v[:,1], v[:,2], color='r', marker='o', label='V1')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=130, azim=-90, roll=0)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_vector2(aa1, aa2, trans1, trans2, v1, v2):
    rotation_matrix1 = tgm.angle_axis_to_rotation_matrix(torch.Tensor(aa1).unsqueeze(0))[0, :3, :3]
    rotation_matrix2 = tgm.angle_axis_to_rotation_matrix(torch.Tensor(aa2).unsqueeze(0))[0, :3, :3]
    vector = np.array([1, 0, 0])
    rotated_vector1 = np.dot(rotation_matrix1, vector)
    rotated_vector2 = np.dot(rotation_matrix2, vector)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(trans1[0], trans1[1], trans1[2], rotated_vector1[0], rotated_vector1[1], rotated_vector1[2], color='r')
    ax.quiver(trans2[0], trans2[1], trans2[2], rotated_vector2[0], rotated_vector2[1], rotated_vector2[2], color='b')
    ax.scatter(trans1[0], trans1[1], trans1[2], color='r', marker='o', label='Point1')
    ax.scatter(trans2[0], trans2[1], trans2[2], color='b', marker='o', label='Point2')
    ax.scatter(v1[:,0], v1[:,1], v1[:,2], color='r', marker='o', label='V1')
    ax.scatter(v2[:,0], v2[:,1], v2[:,2], color='b', marker='o', label='V2')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def transform_mesh(vert_B, trans_A, trans_B, rot_A, rot_B):
    "点Bを点Aの位置、向きに合わせる関数"
    vert_B[:, [0,2]] -= trans_B[:, [0,2]]   # 高さはそのままに平行移動
    # vert_B -= trans_B

    R = np.dot(rot_A, np.linalg.inv(rot_B))
    vert_B = np.dot(R, vert_B.T).T

    # vert_B = np.dot(rot_B.T, vert_B.T).T
    # vert_B = np.dot(rot_A, vert_B.T).T

    vert_B[:, [0,2]] += trans_A[:, [0,2]]
    # vert_B += trans_A
    # vert_B[:, 1] += trans_B[:, 1]
    return vert_B, R


if __name__ == "__main__":

    input_dir       = "./save/humanml_root_key/edit_humanml_root_key_seed10_gp=2.5_(5f_e)_input=rp_alexandra_a_person_walks_and_sits_on_a_chair/sample02_rep00_iter=20"
    mdm_out_npy         = "./save/humanml_root_key/edit_humanml_root_key_seed10_gp=2.5_(5f_e)_input=rp_alexandra_a_person_walks_and_sits_on_a_chair/results.npy"
    posa_rt_path        = "./POSA/save/debug_sample02_rep00_iter=100_sofa/pkl/MPH11/0190_canonical_orient_00.pkl"
    output_ply_dir      = "./save/humanml_root_key/edit_humanml_root_key_seed10_gp=2.5_(5f_e)_input=rp_alexandra_a_person_walks_and_sits_on_a_chair/sample02_rep00_iter=100_converted"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smplx_model_folder = './POSA/POSA_dir/smplx_models'
    neutral_bm_path = './body_models/smplx/SMPLX_NEUTRAL.npz'
    # neutral_bm = BodyModel(bm_fname=neutral_bm_path, num_betas=10).to(device)

    order = 'xyz'

    dtype = torch.float32

    posa_data = joblib.load(posa_rt_path)
    R_cano2posa = np.dot(posa_data['R_fm_orig'][0].detach().cpu().numpy(), R_smpl2scene)
    t_cano2posa = posa_data['t_fm_orig'][0].detach().cpu().numpy()


    mdm_out_data = np.load(mdm_out_npy, allow_pickle=True).item()
    sample_idx = int(os.path.basename(input_dir)[6:8])
    keyframes = mdm_out_data['keyframes'][sample_idx]
    
    # KeyPoseのRt
    key_data = joblib.load(os.path.join(input_dir, "{:04d}.pkl".format(keyframes[0])))
    # key_R = R.from_euler(order, key_data['global_orient'], degrees=False).as_matrix()
    aa_key = key_data['global_orient']
    aa_key[[0,2]] = 0
    R_key = tgm.angle_axis_to_rotation_matrix(torch.Tensor(aa_key).unsqueeze(0))[0, :3, :3]
    t_key = key_data['transl']


    # 各フレームからKeyPoseに変換するRtを保存
    fdata, R_frame2keys, t_frame2keys = [], [], []
    pkl_files, fts, faas  = [], [], []
    for pkl_path in glob.glob(os.path.join(input_dir, "*.pkl")):
        frame = int(os.path.basename(pkl_path)[:-4])
        frame_data = joblib.load(pkl_path)

        # if frame != 0 and frame != 190:
        #     continue

        # 各フレームのRt
        # frame_R = R.from_euler(order, frame_data['global_orient'], degrees=False).as_matrix()
        frame_aa = frame_data['global_orient']
        frame_R = tgm.angle_axis_to_rotation_matrix(torch.Tensor(frame_aa).unsqueeze(0))[0, :3, :3]
        frame_t = frame_data['transl']

        fts.append(frame_t)
        faas.append(frame_aa)

        if frame == 0:
            aa_cano = np.zeros(3)
            R_cano = tgm.angle_axis_to_rotation_matrix(torch.Tensor(aa_cano).unsqueeze(0))[0, :3, :3]
            t_cano = np.zeros([1,3])

            R_key2cano = np.dot(R_cano, np.linalg.inv(R_key))


            ply_path = os.path.join(os.path.dirname(pkl_path), "body_{:04d}.ply".format(frame))
            mesh0 = trimesh.load(ply_path)

            # plot_vector(key_aa, frame_aa, t_key[0], frame_t[0], key_data['vertices'], frame_data['vertices'])

        # 各フレームからKeyPoseに変換するRtを計算
        R_frame2key = np.linalg.inv(R_key) * frame_R.detach().cpu().numpy()
        t_frame2key = t_key - frame_t

        # tmp_dir = output_ply_dir + "_align_190_png_pi2"
        # os.makedirs(tmp_dir, exist_ok=True)
        # plot_vector1(frame_aa, frame_t[0], frame_data['vertices'], save_path=tmp_dir+"/{:04d}.png".format(frame))

        # v = R_frame2key.apply(frame_data['vertices'] + t_frame2key)
        
        # v = transform_mesh2(frame_data['vertices'], t_key, frame_t, R_key, frame_R)

        # tmp_dir = output_ply_dir + "_align_190_2"
        # os.makedirs(tmp_dir, exist_ok=True)
        # mesh = trimesh.Trimesh(vertices=v, faces=mesh0.faces)
        # mesh.export(os.path.join(tmp_dir, "{:04d}.ply".format(frame)))

        R_frame2keys.append(R_frame2key)
        t_frame2keys.append(t_frame2key)

        fdata.append(frame_data)
        pkl_files.append(pkl_path)

        # Open3Dで可視化
        # visualizer = o3d.visualization.Visualizer()
        # visualizer.create_window()
        # p1 = t_key[0]
        # p2 = frame_t[0]
        # visualizer.add_geometry(viz_utils.create_o3d_sphere(p1, radius=0.15))
        # visualizer.add_geometry(viz_utils.create_o3d_sphere(p2, radius=0.15))
        # b1 = viz_utils.create_o3d_mesh_from_np(vertices=frame_data['vertices'], faces=mesh0.faces)
        # b2 = viz_utils.create_o3d_mesh_from_np(vertices=key_data['vertices'], faces=mesh0.faces)
        # visualizer.add_geometry(b1)
        # visualizer.add_geometry(b2)
        # visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
        # visualizer.run()
        # visualizer.destroy_window()
    # exit()

    os.makedirs(output_ply_dir, exist_ok=True)

    for i in tqdm(range(len(fdata))):

        # if i != 0 and i != 190:
        #     continue

        aa_frame = fdata[i]['global_orient']
        R_frame = tgm.angle_axis_to_rotation_matrix(torch.Tensor(aa_frame).unsqueeze(0))[0, :3, :3]
        t_frame = fdata[i]['transl']

        v = np.concatenate([fdata[i]['vertices'], fdata[i]['transl']])
        v_ = copy.deepcopy(v)


        # tmp_dir = output_ply_dir + ""
        # os.makedirs(tmp_dir, exist_ok=True)
        # mesh = trimesh.Trimesh(vertices=v, faces=mesh0.faces)
        # mesh.export(os.path.join(tmp_dir, "{:04d}.ply".format(i)))


        # 各フレームをCanonical Coordinatesに揃える
        # v_cano, R_frame2cano = transform_mesh(v, t_cano, t_frame, R_cano, R_frame)

        v -= t_frame
        v = np.dot(R_key2cano, v.T).T
        v_posa = np.dot(R_cano2posa, v.T).T + t_cano2posa
        v_posa[:, 2] += t_frame[:, 1] - t_key[:, 1]



        # tmp_dir = output_ply_dir + "_canonical"
        # os.makedirs(tmp_dir, exist_ok=True)
        # mesh = trimesh.Trimesh(vertices=v_cano, faces=mesh0.faces)
        # mesh.export(os.path.join(tmp_dir, "canonical_{:04d}.ply".format(i)))

        # pelvis_y = np.zeros([1, 3])
        # # pelvis_y[:, 1] += t_frame[:, 1]

        # # # POSAの座標に変換
        # v_posa = np.dot(R_cano2posa, (v_cano - pelvis_y).T).T + t_cano2posa

        # R_posa = np.dot(R_cano2posa, R_frame2cano)
        # aa_posa = cv2.Rodrigues(R_posa)[0].T[0]

        # tmp_dir = output_ply_dir + "_posa_"
        # os.makedirs(tmp_dir, exist_ok=True)
        # mesh = trimesh.Trimesh(vertices=v_posa, faces=mesh0.faces)
        # mesh.export(os.path.join(tmp_dir, "posa_{:04d}.ply".format(i)))


        # # 元の位置に復元
        t_key2frame = t_frame - t_key
        t_key2frame[0][1] = 0       # 高さは考慮しない
        t_key2frame_on_cano = np.dot(R_key2cano, t_key2frame.T).T   # y軸周りの回転のみ
        t_key2frame_on_posa = np.dot(R_cano2posa, t_key2frame_on_cano.T).T

        v_posa += t_key2frame_on_posa


        tmp_dir = output_ply_dir + "_v_rot"
        os.makedirs(tmp_dir, exist_ok=True)
        mesh = trimesh.Trimesh(vertices=v_posa, faces=mesh0.faces)
        mesh.export(os.path.join(tmp_dir, "v_rot_{:04d}.ply".format(i)))






        # R_posa2cano = np.dot(R_cano, np.linalg.inv(R_posa))
        # R_cano2frame = np.dot(R_frame, np.linalg.inv(R_cano))
        # R_posa2frame = np.dot(R_cano2frame, R_posa2cano)
        # t_cano2frame = t_frame - t_cano


        # tmp_dir = output_ply_dir + "_restore"
        # os.makedirs(tmp_dir, exist_ok=True)
        # mesh = trimesh.Trimesh(vertices=v_restore, faces=mesh0.faces)
        # mesh.export(os.path.join(tmp_dir, "{:04d}.ply".format(i)))


    print()
        

