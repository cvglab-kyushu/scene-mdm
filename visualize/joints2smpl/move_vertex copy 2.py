import os, sys
sys.path.append(os.getcwd())

import glob
import argparse
import copy
import cv2
import joblib
import trimesh
import open3d as o3d
import numpy as np
import torch
import smplx
import torchgeometry as tgm
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from POSA.src import eulerangles, misc_utils
from POSA.src import eulerangles, misc_utils
from visualize.joints2smpl.src import config

# os.environ['PYOPENGL_PLATFORM'] = 'egl'

smplmodel = smplx.create(config.SMPL_MODEL_DIR, 
                         model_type="smplx", gender="neutral", ext="npz",
                         batch_size=1)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mdm_out_dir", type=str, default="()/sample00_rep00_inter=20", help="Output folder for MDM meshed to SMP")
    parser.add_argument("--save_dir", type=str, default="", help="directory to save converted meshes")
    opt = parser.parse_args()

    input_dir = opt.mdm_out_dir
    
    if opt.save_dir == "":
        output_ply_dir = os.path.join(os.path.dirname(input_dir), os.path.basename(input_dir) + "_converted")
    else:
        output_ply_dir = opt.save_dir


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_pca_comps = 6
    body_param_list = [name for name, _ in smplmodel.named_parameters()]


    mdm_out_npy = os.path.join(os.path.dirname(input_dir), "results.npy")
    mdm_out_data = np.load(mdm_out_npy, allow_pickle=True).item()
    sample_idx = int(os.path.basename(input_dir)[6:8])
    keyframe = mdm_out_data['keyframes'][sample_idx]

    R_cano2posa = np.dot(mdm_out_data['R_fm_orig'][sample_idx], R_smpl2scene)
    t_cano2posa = mdm_out_data['t_fm_orig'][sample_idx]


    # KeyPoseのRt
    key_data = joblib.load(os.path.join(input_dir, "{:04d}.pkl".format(keyframe)))
    aa_key = key_data['global_orient']
    aa_key[[0,2]] = 0
    R_key = tgm.angle_axis_to_rotation_matrix(torch.Tensor(aa_key).unsqueeze(0))[0, :3, :3]
    t_key = key_data['transl']


    # Collecting data
    fdata = []
    for pkl_path in glob.glob(os.path.join(input_dir, "*.pkl")):
        frame = int(os.path.basename(pkl_path)[:-4])
        frame_data = joblib.load(pkl_path)

        if frame == 0:
            ply_path = os.path.join(os.path.dirname(pkl_path), "body_{:04d}.ply".format(frame))
            mesh0 = trimesh.load(ply_path)

        fdata.append(frame_data)


    os.makedirs(output_ply_dir, exist_ok=True)


    for i in tqdm(range(len(fdata))):

        if i != keyframe - 1:
            continue

        aa_frame = fdata[i]['global_orient']
        R_frame = tgm.angle_axis_to_rotation_matrix(torch.Tensor(aa_frame).unsqueeze(0))[0, :3, :3]
        t_frame = fdata[i]['transl']
        v_frame = np.concatenate([fdata[i]['vertices'], fdata[i]['transl']])

        torch_param = {}
        for key in fdata[i].keys():
            if key in body_param_list:
                torch_param[key] = torch.tensor(fdata[i][key], dtype=torch.float32).to(device)
                if key == "body_pose" and torch_param[key].shape[1] > 63:
                    torch_param[key] = torch_param[key][:,3:]

        torch_param['betas'] = torch_param['betas'][:, :10]
        torch_param['left_hand_pose'] = torch_param['left_hand_pose'][:, :num_pca_comps]
        torch_param['right_hand_pose'] = torch_param['right_hand_pose'][:, :num_pca_comps]
        torch_param['transl'][:, [0,2]] -= torch_param['transl'][:, [0,2]]

        smplmodel.reset_params(**torch_param)
        smplmodel_output = smplmodel(return_verts=True)
        pelvis = smplmodel_output.joints[:, 0, :].reshape(1, 3)
        v_frame = smplmodel_output.vertices.squeeze()


        # 各フレームをKeyPoseがCanonicalになる分回転
        v_frame = np.dot(np.linalg.inv(R_key), (v_frame - pelvis).T.detach().cpu().numpy()).T

        # POSAの座標に変換
        v_posa = np.dot(R_cano2posa, v_frame.T).T + t_cano2posa

        # POSAの座標変換の確認
        # tmp_dir = output_ply_dir + "_posa_"
        # os.makedirs(tmp_dir, exist_ok=True)
        # mesh = trimesh.Trimesh(vertices=v_posa, faces=mesh0.faces)
        # mesh.export(os.path.join(tmp_dir, "posa_{:04d}.ply".format(i)))


        # # 各フレームをKeyPoseに対して元の位置に復元
        t_key2frame = t_frame - t_key
        t_key2frame_on_cano = np.dot(np.linalg.inv(R_key), t_key2frame.T).T   # y軸周りの回転のみ
        t_key2frame_on_posa = np.dot(R_cano2posa, t_key2frame_on_cano.T).T
        v_posa += t_key2frame_on_posa

        # 保存
        tmp_dir = output_ply_dir
        os.makedirs(tmp_dir, exist_ok=True)
        mesh = trimesh.Trimesh(vertices=v_posa, faces=mesh0.faces)
        mesh.export(os.path.join(tmp_dir, "body_{:04d}.ply".format(i)))


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
        
    print("Output to {}".format(output_ply_dir))
        

