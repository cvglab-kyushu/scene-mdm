import copy
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

HML_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]

NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 22 SMPLH body joints

HML_LOWER_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot',]]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]

# NUM_HML_FEATS = 263
# NUM_HML_FEATS = 159
NUM_HML_FEATS = 69

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS-1))
HML_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False] * 4))
HML_ROOT_HORIZONTAL_MASK = np.concatenate(([True]*(1+2) + [False],
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(3)),
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(6)),
                                np.zeros_like(HML_ROOT_BINARY.repeat(3)),
                                [False] * 4))
HML_SMPLX_ROOT_ORIENT_MASK = np.zeros_like(HML_ROOT_MASK)
HML_SMPLX_ROOT_ORIENT_MASK[0:4] = True
HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                     HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                     [True]*4))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK


HML_TRAJ_MASK = np.zeros(266)
HML_TRAJ_MASK[4:7] = 1
HML_TRAJ_MASK_Y = copy.deepcopy(HML_TRAJ_MASK)
HML_TRAJ_MASK_Y[5] = 0

lhand_id = HML_JOINT_NAMES.index('left_wrist')-1
rhand_id = HML_JOINT_NAMES.index('right_wrist')-1

def expand_mask(mask, shape):
    """
    expands a mask of shape (num_feat, seq_len) to the requested shape (usually, (batch_size, num_feat, 1, seq_len))
    """
    _, num_feat, _, _ = shape
    return np.ones(shape) * mask.reshape((1, num_feat, 1, -1))

def get_joints_mask(join_names, num_feat):
    joins_mask = np.array([joint_name in join_names for joint_name in HML_JOINT_NAMES])
    # mask = np.concatenate(([False]*(1+2+1),
    mask = np.concatenate(([False]*(4+3),
                                joins_mask[1:].repeat(3),
                                np.zeros_like(joins_mask[1:].repeat(6)),
                                np.zeros_like(joins_mask.repeat(3)),
                                [False] * 4))
    mask = mask[:num_feat]
    return mask

def get_batch_joint_mask(shape, joint_names, num_feat):
    return expand_mask(get_joints_mask(joint_names, num_feat), shape)

def get_in_between_mask(shape, lengths, prefix_end, suffix_end):
    mask = np.ones(shape)  # True means use gt motion
    for i, length in enumerate(lengths):
        start_idx, end_idx = int(prefix_end * length), int(suffix_end * length)
        mask[i, :, :, start_idx: end_idx] = 0  # do inpainting in those frames
    return mask

def get_prefix_mask(shape, prefix_length=20):
    _, num_feat, _, seq_len = shape
    prefix_mask = np.concatenate((np.ones((num_feat, prefix_length)), np.zeros((num_feat, seq_len - prefix_length))), axis=-1)
    return expand_mask(prefix_mask, shape)

def get_inpainting_mask(mask_name, shape, length=None, **kwargs):
    mask_names = mask_name.split(',')

    NUM_HML_FEATS = shape[1]
    
    mask = np.zeros(shape)
    if 'only_text' in mask_names:
        kfs = []
        for i in range(len(length)):
            leng = length[i] if length[i] < 196 else 195
            if kwargs['keyframes'] is None:
                # Random masking
                # keyframes = np.random.choice(range(1, leng-1), random.randint(0, 3), replace=False)
                keyframes = np.random.choice(range(1, leng-1), random.randint(1, 1), replace=False)
            kfs.append(keyframes)
        return mask, []

    if 'in_between' in mask_names:
        mask = np.maximum(mask, get_in_between_mask(shape, **kwargs))
    
    elif 'root' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_ROOT_MASK, shape))
    
    elif 'root_horizontal' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_ROOT_HORIZONTAL_MASK, shape))

        kfs = []
        for i in range(len(length)):
            leng = length[i] if length[i] < 196 else 195
            if kwargs['keyframes'] is None:
                # Random masking
                keyframes = np.random.choice(range(1, leng-1), random.randint(0, 3), replace=False)
            kfs.append(keyframes)

    elif mask_names[0] == 'key':
        # first frame masking
        mask[:, :, :, 0] = np.ones(mask[:, :, :, 0].shape)

        # random frame (0~3) masking
        # for i in range(len(length)):
        #     leng = length[i] if length[i] < 196 else 195
        #     keyframes = np.random.choice(leng, random.randint(0, 3), replace=False)
        #     mask[i, :, :, keyframes] = np.ones(mask[i, :, :, leng].shape)

    elif 'root_key_y' in mask_names[0] or 'traj_key_y' in mask_names[0]:
        mask = np.maximum(mask, expand_mask(HML_TRAJ_MASK_Y[:NUM_HML_FEATS], shape))
        kfs = []
        for i in range(len(length)):
            leng = length[i] if length[i] < 196 else 195

            if kwargs['keyframes'] is None:
                # Random masking
                # keyframes = np.random.choice(range(1, leng-1), random.randint(10, 15), replace=False)
                # keyframes = np.random.choice(range(1, leng-1), random.randint(1, 3), replace=False)
                # keyframes = np.random.choice(range(1, leng-4), random.randint(1, 1), replace=False)
                
                # keyframes = np.arange(leng-11, leng-1)

                # keyframes = []
                
                # # last frame
                keyframes = [leng - 1]
                # first frame mask
                # keyframes = [0]
                # keyframes = [int(leng/2) - 20, int(leng/2), int(leng/2) + 20]
                # keyframes = [int(leng/2)]
            else:
                keyframes = kwargs['keyframes']
            kfs.append(keyframes)
            mask[i, :, :, keyframes] = np.ones(mask[i, :, :, leng].shape)
        return np.maximum(mask, get_batch_joint_mask(shape, mask_names, NUM_HML_FEATS)), kfs
        

    elif 'root_key' in mask_names[0] or 'traj_key' in mask_names[0]:
        mask = np.maximum(mask, expand_mask(HML_TRAJ_MASK[:NUM_HML_FEATS], shape))
        kfs = []
        for i in range(len(length)):
            leng = length[i] if length[i] < 196 else 195
            # Random masking
            # keyframes = np.random.choice(range(1, leng-1), random.randint(0, 3), replace=False)
            # # 5 frame before last frame
            keyframes = [leng - 5]
            # first frame mask
            # # keyframes = [1]
            mask[i, :, :, keyframes] = np.ones(mask[i, :, :, leng].shape)
            kfs.append(keyframes)
        return np.maximum(mask, get_batch_joint_mask(shape, mask_names, NUM_HML_FEATS)), kfs


    elif 'waypoints' in mask_names[0]:
        # mask the keypose and some waypoints
        mask = np.maximum(mask, expand_mask(np.zeros(266), shape))
        kfs = []
        for i in range(len(length)):
            leng = length[i] if length[i] < 196 else 195
            waypoints = np.random.choice(range(1, leng-1), random.randint(0, 5), replace=False)
            waypoints = np.concatenate([waypoints, np.array([0, leng])])
            mask[i,4,0,waypoints] = 1
            mask[i,6,0,waypoints] = 1

            # # Random frame
            # keyframes =  np.random.choice(range(1, leng-1), random.randint(1, 1), replace=False)
            keyframes =  [leng - 5]
            kfs.append(keyframes)
            mask[i, :, :, keyframes] = np.ones(mask[i, :, :, leng].shape)
        return np.maximum(mask, get_batch_joint_mask(shape, mask_names, NUM_HML_FEATS)), kfs

    elif 'key_joint' in mask_names[0]:
        # mask the left or right or both joint at the final frame
        mask = np.maximum(mask, expand_mask(HML_TRAJ_MASK_Y[:NUM_HML_FEATS], shape))
        kfs = []
        for i in range(len(length)):
            leng = length[i] if length[i] <= 196 else 196
            rand = random.randint(0,2)
            if rand == 0:
                mask[i,7+lhand_id*3:7+lhand_id*3+3,0,leng-1] = 1
            elif rand == 1:
                mask[i,7+rhand_id*3:7+rhand_id*3+3,0,leng-1] = 1
            else:
                mask[i,7+lhand_id*3:7+lhand_id*3+3,0,leng-1] = 1
                mask[i,7+rhand_id*3:7+rhand_id*3+3,0,leng-1] = 1
            keyframes =  [leng-1]
            kfs.append(keyframes)
        return np.maximum(mask, get_batch_joint_mask(shape, mask_names, NUM_HML_FEATS)), kfs

    if 'root_orient_key' in mask_names[0]:
        mask = np.maximum(mask, expand_mask(HML_SMPLX_ROOT_ORIENT_MASK[:NUM_HML_FEATS], shape))
        for i in range(len(length)):
            leng = length[i] if length[i] < 196 else 195
            # # Random masking
            # keyframes = np.random.choice(leng, random.randint(0, 3), replace=False)
            # 5 frame before last frame
            keyframes = [leng - 5]
            # # first frame mask
            # keyframes = [1]

            mask[i, :, :, keyframes] = np.ones(mask[i, :, :, leng].shape)

    if 'root_goal' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_ROOT_HORIZONTAL_MASK, shape))
        for i in range(len(length)):
            leng = length[i] if length[i] < 196 else 195
            mask[i, :, :, leng] = np.ones(mask[i, :, :, leng].shape)

    if 'prefix' in mask_names:
        mask = np.maximum(mask, get_prefix_mask(shape, **kwargs))

    if 'upper_body' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_UPPER_BODY_MASK, shape))
    
    if 'lower_body' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_LOWER_BODY_MASK, shape))
    
    return np.maximum(mask, get_batch_joint_mask(shape, mask_names, NUM_HML_FEATS))