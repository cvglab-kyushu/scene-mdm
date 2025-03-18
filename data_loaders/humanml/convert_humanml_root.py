import sys
sys.path.append("./")

import os
import glob
import torch
import numpy as np
from tqdm import tqdm

from data_loaders.humanml.scripts.motion_process import convert_humanml_root

if __name__ == "__main__":
    data_dir = "../motion-diffusion-model/dataset/HumanML3D/new_joint_vecs"
    save_dir = "../motion-diffusion-model/dataset/HumanML3D/new_joint_vecs_266"


    # file_list = glob.glob(data_dir + "/*.npy")
    # os.makedirs(save_dir, exist_ok=True)
    # for motion_path in tqdm(file_list):
    #     motion = np.load(motion_path)

    #     motion = torch.tensor(motion).unsqueeze(0).unsqueeze(0)

    #     out = convert_humanml_root(motion)
    #     out = out.detach().cpu().numpy()[0][0]

    #     save_path = save_dir + '/' + os.path.basename(motion_path)
    #     np.save(save_path, out)


    # Save mean and std files
    save_mean_path =  "../motion-diffusion-model/dataset/HumanML3D/Mean_266.npy"
    save_std_path =  "../motion-diffusion-model/dataset/HumanML3D/Std_266.npy"
    file_list = glob.glob(save_dir + "/*.npy")
    motions = []
    for motion_path in tqdm(file_list):
        motions.append(np.load(motion_path))
    
    poses = np.concatenate(motions)
    mean = np.mean(poses, 0)
    std = np.std(poses, 0)
    std[np.where(std==0)[0]] = 1
    np.save(save_mean_path, mean)
    np.save(save_std_path, std)




