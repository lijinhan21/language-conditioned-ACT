import math
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import h5py
from tqdm import tqdm
import time
import yaml
import pickle
import torch
import cv2
from collections import deque
import argparse
import sys

from einops import rearrange

from pathlib import Path
current_dir = Path(__file__).parent.resolve()
LOG_DIR = (current_dir / 'logs/').resolve()

from imitate_episodes import make_policy, load_ckpt, make_config
from utils import joint_state_26_to_56_real, joint_state_56_to_26

from gr1_interface.gr1_control.gr1_client import gr1_interface
from gr1_interface.gr1_control.utils.variables import (
    finger_joints,
    name_to_limits,
    name_to_sign,
    name_to_urdf_idx,
)

from deoxys_vision.utils.camera_utils import assert_camera_ref_convention, get_camera_info
from deoxys_vision.networking.camera_redis_interface import CameraRedisSubInterface

def get_norm_stats(data_path):
    with open(data_path, "rb") as f:
        norm_stats = pickle.load(f)
    return norm_stats

def load_policy(config):
    ckpt_dir = config['ckpt_dir']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    
    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    
    policy, ckpt_name, epoch = load_ckpt(policy, ckpt_dir, config['resume_ckpt'])
    
    return policy

def normalize_input(state, agentview_rgb, norm_stats, last_action_data=None):
    agentview_rgb = rearrange(agentview_rgb, 'h w c -> c h w')
    image_data = torch.from_numpy(np.stack([agentview_rgb], axis=0)) / 255.0
    qpos_data = (torch.from_numpy(state) - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
    image_data = image_data.view((1, 1, 3, 224, 224)).float().to(device='cuda')
    state_dim = len(norm_stats["qpos_mean"])
    qpos_data = qpos_data.view((1, state_dim)).float().to(device='cuda') # TODO: change 13 to state_dim

    if last_action_data is not None:
        last_action_data = torch.from_numpy(last_action_data).to(device='cuda').view((1, -1)).to(torch.float)
        qpos_data = torch.cat((qpos_data, last_action_data), dim=1)
    return (qpos_data, image_data)

def cut_img(img):
    assert img.shape[0] == 720 and img.shape[1] == 1280
    lower_img = img[180:, :, :]
    return lower_img

def resize_img(img):
    resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return resized_img

def raw_image_preprocess(image, crop=False):
    if crop:
        img_lower = cut_img(image)
    else:
        img_lower = image.copy()
    img_final = resize_img(img_lower)
    return img_final

def merge_act(actions_for_curr_step, k = 0.01):
    actions_populated = np.all(actions_for_curr_step != 0, axis=1)
    actions_for_curr_step = actions_for_curr_step[actions_populated]

    exp_weights = np.exp(-k * np.arange(actions_for_curr_step.shape[0]))
    exp_weights = (exp_weights / exp_weights.sum()).reshape((-1, 1))
    raw_action = (actions_for_curr_step * exp_weights).sum(axis=0)

    return raw_action

finger_joint_idxs = [name_to_urdf_idx[j] for j in finger_joints]
finger_joint_max = np.array([name_to_limits[j][1] for j in finger_joints])
finger_joint_min = np.array([name_to_limits[j][0] for j in finger_joints])

def process_urdf_joints(joints, shoulder_offset=0):
    joints = joints * 180.0 / np.pi
    sign_array = np.ones(56)
    # body joints
    for name_idx in name_to_sign:
        sign_array[name_to_urdf_idx[name_idx]] = name_to_sign[name_idx]
    joints[7] += shoulder_offset
    joints[26] -= shoulder_offset
    joints *= sign_array
    
    # hand joints
    hand_joints = joints[finger_joint_idxs].copy()
    hand_joints_limited = np.clip(hand_joints, finger_joint_min, finger_joint_max)
    hand_joints_rel = (hand_joints_limited - finger_joint_min) / (
        finger_joint_max - finger_joint_min
    )
    hand_joints_int = (1.0 - hand_joints_rel) * 1000

    hand_joints_int = np.clip(hand_joints_int, 0, 1000).astype(int)
    # switch left right finger control
    hand_joints_int_reorderd = hand_joints_int[[5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6]]

    return joints, hand_joints_int_reorderd

def run_interpolation(start_pos, end_pos, gr1, steps=50, state_saver=None):
    for i in range(steps):
        start_time = time.time_ns()
        
        q = start_pos + (end_pos - start_pos) * (i / steps)
        body_joints, hand_joints = process_urdf_joints(q)
        
        gr1.control(arm_cmd=body_joints, hand_cmd=hand_joints, terminate=False)
        if state_saver is not None:
            state_saver.add_state(body_joints.copy())

        end_time = time.time_ns()
        time.sleep(
            gr1.interval - np.clip(((end_time - start_time) / (10**9)), 0, gr1.interval)
        )

def interpolate_to_start_pos(gr1, steps=50, state_saver=None, lr=None):
    init_pos = np.zeros(56)

    T_pos = np.zeros(56)
    if lr is not None:
        T_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if lr == 'R' else 1) * 10 
    T_pos[name_to_urdf_idx['joint_head_pitch']] = 19 
    T_pos[name_to_urdf_idx['l_shoulder_roll']] = -90 
    T_pos[name_to_urdf_idx['r_shoulder_roll']] = 90 
    T_pos = T_pos / 180 * np.pi

    L_pos = np.zeros(56)
    if lr is not None:
        L_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if lr == 'R' else 1) * 10
    # L_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if args.lr == 'R' else 1) * 10
    L_pos[name_to_urdf_idx['joint_head_pitch']] = 19
    L_pos[name_to_urdf_idx['l_shoulder_roll']] = -90
    L_pos[name_to_urdf_idx['r_shoulder_roll']] = 90
    L_pos[name_to_urdf_idx['l_elbow_pitch']] = 90
    L_pos[name_to_urdf_idx['r_elbow_pitch']] = -90
    L_pos = L_pos / 180 * np.pi

    # Interpolate to T-pos
    run_interpolation(init_pos, T_pos, gr1, steps, state_saver)
    # Interpolate to L-pos
    run_interpolation(T_pos, L_pos, gr1, steps, state_saver)
    return L_pos

def interpolate_to_end_pos(current_pos, gr1, steps=50, state_saver=None):

    L_pos = np.zeros(56)
    # L_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if args.lr == 'R' else 1) * 10
    L_pos[name_to_urdf_idx['joint_head_pitch']] = 19
    L_pos[name_to_urdf_idx['l_shoulder_roll']] = -90
    L_pos[name_to_urdf_idx['r_shoulder_roll']] = 90
    L_pos[name_to_urdf_idx['l_elbow_pitch']] = 90
    L_pos[name_to_urdf_idx['r_elbow_pitch']] = -90
    L_pos = L_pos / 180 * np.pi

    T_pos = np.zeros(56)
    # T_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if args.lr == 'R' else 1) * 10 
    T_pos[name_to_urdf_idx['joint_head_pitch']] = 19 
    T_pos[name_to_urdf_idx['l_shoulder_roll']] = -90 
    T_pos[name_to_urdf_idx['r_shoulder_roll']] = 90 
    T_pos = T_pos / 180 * np.pi

    run_interpolation(current_pos, L_pos, gr1, steps, state_saver)
    run_interpolation(L_pos, T_pos, gr1, steps, state_saver)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--qpos_noise_std', action='store', default=0, type=float, help='lr', required=False)

    parser.add_argument('--backbone', action='store', type=str, default='resnet18', help='visual backbone, choose from resnet18, resnet34, dino_v2', required=False)
    parser.add_argument('--state_dim', action='store', type=int, default=13, help='state_dim', required=False)
    parser.add_argument('--action_dim', action='store', type=int, default=13, help='action_dim', required=False)
    
    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--save_jit', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--resumeid', action='store', default="", type=str, help='resume id', required=False)
    parser.add_argument('--resume_ckpt', action='store', type=str, help='resume ckpt', required=True)
    parser.add_argument('--task-name', action='store', type=str, help='task name', required=True)
    parser.add_argument('--exptid', action='store', type=str, help='experiment id', required=True)
    parser.add_argument('--dataset-path', action='store', type=str, help='path_to_hdf5_dataset', required=True)
    
    parser.add_argument('--saving-interval', action='store', type=int, default=5000, help='saving interval', required=False)
    args = vars(parser.parse_args())

    config = make_config(args)
    
    ckpt_dir = config['ckpt_dir']
    norm_stat_path = Path(ckpt_dir) / "dataset_stats.pkl"
    
    temporal_agg = True
    action_dim = args['action_dim']

    chunk_size = args['chunk_size']
    device = "cuda"
    
    timestamps = 300 # max length of an episode

    norm_stats = get_norm_stats(norm_stat_path)
    policy = load_policy(config)
    policy.cuda()
    policy.eval()

    history_stack = 0
    last_action_queue = None
    last_action_data = None
    
    # start GR1 interface
    gr1 = gr1_interface(
        "10.42.0.21", pub_port_arm=5555, pub_port_hand=6666, sub_port=5556, rate=40
    )   
    print("start gr1 interface!")

    # start redis client
    assert_camera_ref_convention('rs_0')
    camera_info = get_camera_info('rs_0')
    camera_id = camera_info.camera_id
    cr_interface = CameraRedisSubInterface(redis_host="localhost", camera_info=camera_info, use_depth=True)
    cr_interface.start()
    
    # ---

    if temporal_agg:
        all_time_actions = np.zeros([timestamps, timestamps+chunk_size, action_dim])
    else:
        num_actions_exe = chunk_size
    
    try:

        num_exp = 16
        success_count = 0

        for exp_id in range(num_exp):
            
            print("start new episode!", exp_id + 1, "/", num_exp)

            output = None

            # interpolate to start pos
            start_pos = interpolate_to_start_pos(gr1, steps=50, lr=None) 
            input("Please Adjust the positions of objects. Afterwards, press Enter to start...")

            last_state = start_pos.copy()
                
            for t in tqdm(range(timestamps)):
                if history_stack > 0:
                    last_action_data = np.array(last_action_queue)

                state = joint_state_56_to_26(last_state.copy())
                
                imgs = cr_interface.get_img() # getting img from redis server
                agentview_rgb = cv2.cvtColor(imgs['color'], cv2.COLOR_BGR2RGB)
                agentview_rgb = raw_image_preprocess(agentview_rgb, crop=True)
                
                data = normalize_input(state, agentview_rgb, norm_stats, last_action_data)

                if temporal_agg: # Must be true
                    output = policy(*data)[0].detach().cpu().numpy() # (1,chuck_size,action_dim)
                    all_time_actions[[t], t:t+chunk_size] = output
                    act = merge_act(all_time_actions[:, t])
                    
                act = act * norm_stats["action_std"] + norm_stats["action_mean"]

                urdf_action = joint_state_26_to_56_real(act)

                # convert urdf_action to real robot action, and send it to the robot
                run_interpolation(last_state, urdf_action, gr1, 2, state_saver=None)

                last_state = urdf_action.copy()
            
            interpolate_to_end_pos(last_state, gr1, steps=50)

            str = input("Success or not?[y/n]")
            if str == 'y':
                print("Success!")
                success_count += 1
            else:
                print("Failed!")

            input("Press Enter to start next episode...")

        print("Success rate:", success_count / num_exp)

        # terminate gr1
        gr1.control(terminate=True)
        gr1.close_threads()

    except KeyboardInterrupt:
        
        gr1.control(terminate=True)
        gr1.close_threads()
        
        exit()