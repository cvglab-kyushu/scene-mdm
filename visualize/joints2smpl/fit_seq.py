from __future__ import print_function, division
import argparse
import torch
import os,sys
from os import walk, listdir
from os.path import isfile, join
import numpy as np
import joblib
import smplx
import trimesh
import h5py
from tqdm import tqdm
import glob
import time
import copy
import open3d as o3d

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from smplify import SMPLify3D
import config
from POSA.src import posa_utils, eulerangles, viz_utils, misc_utils, data_utils, opt_utils


# parsing argmument
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1,
                    help='input batch size')
parser.add_argument('--num_smplify_iters', type=int, default=100,
                    help='num of smplify iters')
parser.add_argument('--cuda', type=bool, default=False,
                    help='enables cuda')
parser.add_argument('--flip', type=bool, default=False,
                    help='enables cuda')
parser.add_argument('--gpu_ids', type=int, default=0,
                    help='choose gpu ids')
parser.add_argument('--num_joints', type=int, default=22,
                    help='joint number')
parser.add_argument('--joint_category', type=str, default="AMASS_smplx",
                    help='use correspondence')
parser.add_argument('--fix_foot', type=str, default="False",
                    help='fix foot or not')
parser.add_argument('--smpl_mode', type=str, default="smplx",
                    help='smpl or smplx')
parser.add_argument("--input_path", type=str, required=True, 
					help='stick figure mp4 file to be rendered.')

opt = parser.parse_args()
print(opt)

# ---load predefined something
device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")
print(config.SMPL_MODEL_DIR)
smplmodel = smplx.create(config.SMPL_MODEL_DIR, 
                         model_type=opt.smpl_mode, gender="neutral", ext="npz",
                         batch_size=opt.batchSize).to(device)

num_joints = 22 if opt.smpl_mode == "smpl" else 21
num_init_pose = 72 if opt.smpl_mode == "smpl" else 66

# ## --- load the mean pose as original ---- 
smpl_mean_file = config.SMPL_MEAN_FILE

file = h5py.File(smpl_mean_file, 'r')
init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).float()
if opt.smpl_mode == "smplx":
	init_mean_pose = init_mean_pose[:, :num_init_pose]
# init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).float()
init_mean_shape = torch.zeros([1,10]).to(device)
cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).to(device)
#
pred_pose = torch.zeros(opt.batchSize, num_init_pose).to(device)

sample_beta = np.array([[-0.7493872 , -0.51256967,  1.1641908 , -0.75547874, -2.1708083 ,
       -0.03381369, -0.38829893,  0.45629647,  0.7177078 ,  0.04079496]])

pred_betas = torch.FloatTensor(sample_beta).to(device)
pred_cam_t = torch.zeros(opt.batchSize, 3).to(device)
keypoints_3d = torch.zeros(opt.batchSize, num_joints, 3).to(device)

# # #-------------initialize SMPLify
smplify = SMPLify3D(smplxmodel=smplmodel,
                    batch_size=opt.batchSize,
                    joints_category=opt.joint_category,
					num_iters=opt.num_smplify_iters,
                    device=device)
#print("initialize SMPLify3D done!")



input_path = opt.input_path
parsed_name = os.path.basename(input_path).replace('.mp4', '').replace('sample', '').replace('rep', '')
sample_i, rep_i = [int(e) for e in parsed_name.split('_')]
npy_path = os.path.join(os.path.dirname(input_path), 'results.npy')
out_npy_path = input_path.replace('.mp4', '_smpl_params.npy')
assert os.path.exists(npy_path)
results_dir = input_path.replace('.mp4', '_iter={}'.format(opt.num_smplify_iters))
os.makedirs(results_dir, exist_ok=True)

motions = np.load(npy_path, allow_pickle=True)
motions = motions[None][0]
bs, njoints, nfeats, nframes = motions['motion'].shape
opt_cache = {}
total_num_samples = motions['num_samples']
absl_idx = rep_i*total_num_samples + sample_i
num_frames = int(motions['lengths'][absl_idx])
keyframes = motions['keyframes'][sample_i]

data = motions['motion'][absl_idx].transpose(2, 0, 1)
data = data[:num_frames]


# # reverse y and z
# data[:,:,0] *= -1
# data[:,:,1] *= -1
# data[:,:,2] *= -1

# # --- load data ---
# data = np.load(opt.data_folder + "/" + purename + ".npy")  # [nframes, njoints, 3]

# # debug
# import copy
# from human_body_prior.body_model.body_model import BodyModel
# neutral_bm_path = './body_models/smplx/SMPLX_NEUTRAL.npz'
# neutral_bm = BodyModel(bm_fname=neutral_bm_path, num_betas=10).cuda()
# tdata = torch.Tensor(copy.deepcopy(data))
# tdata = tdata - tdata[:, 0:1, :]
# trans = tdata[:, 0, :].cuda()
# pose_body = tdata[:, 1:, :].reshape(tdata.shape[0], -1).cuda()
# betas = torch.zeros(tdata.shape[0], 10).cuda()
# body = neutral_bm(pose_body=pose_body, betas=betas, trans=trans)
# mesh = trimesh.Trimesh(vertices=body.v[0].cpu().numpy(), faces=body.f.cpu().numpy())
# save_dir = "./save/debug/tmp_obj"
# os.makedirs(save_dir, exist_ok=True)
# for i in range(body.v.shape[0]):
# 	mesh.vertices = body.v[i].cpu().numpy()
# 	mesh.export(os.path.join(save_dir, "body_{}.obj".format(str(i).zfill(4))))


# flip joints
if opt.flip:
	flip_joints = [[1,2], [4,5], [7,8], [10,11], [13,14], [16,17], [18,19], [20,21]]
	flipped_keypoints = copy.deepcopy(data)
	for a, b in flip_joints:
		flipped_keypoints[:, [a, b], :] = flipped_keypoints[:, [b, a], :]
	data = copy.deepcopy(flipped_keypoints)


dir_save = results_dir

# run the whole seqs
num_seqs = data.shape[0]

vertices, poses, betas = [], [], []
for idx in tqdm(range(num_seqs)):

	# start_time = time.time() # Delete this 

	# if idx < 188:
	# 	continue

	#print(idx)
	pkl_save_path = dir_save + "/" + "%04d"%idx + ".pkl"
	mesh_save_path = dir_save + "/body_%04d"%idx + ".ply"

	# if os.path.exists(pkl_save_path):
	# 	continue

	if opt.smpl_mode == "smpl":
		joints3d = data[idx] #*1.2 #scale problem [check first]
	elif opt.smpl_mode == "smplx":
		joints3d = data[idx][:21]

	keypoints_3d[0, :, :] = torch.Tensor(joints3d).to(device).float()

	if idx == 0:
		pred_betas[0, :] = init_mean_shape
		pred_pose[0, :] = init_mean_pose
		pred_cam_t[0, :] = cam_trans_zero
	else:
		data_param = joblib.load(dir_save + "/" + "%04d"%(idx-1) + ".pkl")
		pred_betas[0, :] = torch.from_numpy(data_param['betas']).unsqueeze(0).float()
		pred_pose[0, :] = torch.from_numpy(data_param['body_pose']).unsqueeze(0).float()
		pred_cam_t[0, :] = torch.from_numpy(data_param['transl']).unsqueeze(0).float()
		
	if opt.joint_category =="AMASS" or opt.joint_category =="AMASS_smplx":
		confidence_input =  torch.ones(num_joints)
		# make sure the foot and ankle
		if opt.fix_foot == True:
			confidence_input[7] = 1.5
			confidence_input[8] = 1.5
			confidence_input[10] = 1.5
			confidence_input[11] = 1.5
	else:
		confidence_input =  torch.ones(num_joints)
		print("Such category not settle down!")
	
	# before_simplify_time = time.time() # Delete this 


	# ----- from initial to fitting -------
	new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
		new_opt_cam_t, new_opt_joint_loss = smplify(
												pred_pose.detach(),
												pred_betas.detach(),
												pred_cam_t.detach(),
												keypoints_3d,
												conf_3d=confidence_input.to(device),
												seq_ind=idx
												)
	
	# after_simplify_time = time.time() # Delete this 

	# # -- save the results to ply---
	new_opt_cam_t = new_opt_cam_t[0]
	outputp = smplmodel(betas=new_opt_betas, global_orient=new_opt_pose[:, :3], body_pose=new_opt_pose[:, 3:],
						transl=new_opt_cam_t, return_verts=True)
	
	# after_smplmodel_time = time.time() # Delete this 

	vertex = outputp.vertices.detach().cpu().numpy().squeeze()
	mesh_p = trimesh.Trimesh(vertices=vertex, faces=smplmodel.faces, process=False)
	mesh_p.export(mesh_save_path)

	vertices.append(vertex)
	poses.append(new_opt_pose.detach().cpu().numpy().squeeze().reshape((22, 3)))
	betas.append(new_opt_betas.detach().cpu().numpy().squeeze())
	
	# save the pkl
	param = {}
	param['global_orient'] = outputp.global_orient[0].detach().cpu().numpy()
	param['betas'] = new_opt_betas.detach().cpu().numpy()
	param['body_pose'] = new_opt_pose.detach().cpu().numpy()
	param['vertices'] = outputp.vertices.detach().cpu().numpy().squeeze()
	param['transl'] = new_opt_cam_t[0].detach().cpu().numpy()
	# param['transl'] = joints3d[0].reshape(1, 3)
	param['gender'] = "neutral"
	param['left_hand_pose'] = outputp.left_hand_pose.detach().cpu().numpy()
	param['right_hand_pose'] = outputp.right_hand_pose.detach().cpu().numpy()
	joblib.dump(param, pkl_save_path, compress=3)

	# after_save_time = time.time() # Delete this 

	# print()
	# print("Preprocess	: {}".format(before_simplify_time - start_time))
	# print("Simplify	: {}".format(after_simplify_time - before_simplify_time))
	# print("Smplmodel	: {}".format(after_smplmodel_time - after_simplify_time))
	# print("Save Mesh	: {}".format(after_save_time - after_smplmodel_time))
	# print()


data = {
		"vertices": np.array(vertices),
		"poses": np.array(poses),
		"betas": np.array(betas),
	}
np.save(dir_save + "/mesh_motion.npy", data)