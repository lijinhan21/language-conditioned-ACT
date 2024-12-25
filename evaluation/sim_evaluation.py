import math
import numpy as np
import matplotlib.pyplot as plt

from transformers import CLIPModel, CLIPTokenizer

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
import os

from einops import rearrange

from pathlib import Path
current_dir = Path(__file__).parent.resolve()
LOG_DIR = (current_dir / '../act/logs/').resolve()

import init_path
from act.imitate_episodes import make_policy, load_ckpt, make_config
from evaluation.replay_data import Player

def get_norm_stats(data_path):
    with open(data_path, "rb") as f:
        norm_stats = pickle.load(f)
    return norm_stats

def load_policy_jit(policy_path, device):
    policy = torch.jit.load(policy_path, map_location=device)
    return policy

def load_policy(config):
    ckpt_dir = config['ckpt_dir']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    
    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    
    policy, ckpt_name, epoch = load_ckpt(policy, ckpt_dir, config['resume_ckpt'])
    
    return policy

def normalize_input(state, lang, images, norm_stats, CLIP_tokenizer, bsz=1):
    
    all_image_data = torch.zeros((bsz, len(images[0]), 3, 128, 128), device='cuda')
    state_dim = len(norm_stats["qpos_mean"])
    all_qpos_data = torch.zeros((bsz, state_dim), device='cuda')
    all_lang_data = {
        'input_ids': torch.zeros((bsz, 1, 25), device='cuda', dtype=torch.int64),
        'attention_mask': torch.zeros((bsz, 1, 25), device='cuda', dtype=torch.int64),
    }
    
    for i in range(bsz):
        
        all_cam_images = []
        for image in images[i]:
            all_cam_images.append(rearrange(image, 'h w c -> c h w'))
        image_data = torch.from_numpy(np.stack([all_cam_images], axis=0)) / 255.0
        image_data = image_data.view((1, len(all_cam_images), 3, 128, 128)).float().to(device='cuda') 
        all_image_data[i] = image_data
        
        state_dim = len(norm_stats["qpos_mean"])
        qpos_data = (torch.from_numpy(state[i]) - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
        qpos_data = qpos_data.view((1, state_dim)).float().to(device='cuda')
        all_qpos_data[i] = qpos_data
        
        lang_data = CLIP_tokenizer(
                lang, 
                padding='max_length', 
                truncation=True, 
                max_length=25,
                return_tensors="pt"
        )
        for key in lang_data.keys():
            lang_data[key] = lang_data[key].cuda()
            all_lang_data[key][i][0] = lang_data[key]
    
    # print("shapes:", all_qpos_data.shape, all_image_data.shape)
    # for key in all_lang_data.keys():
    #     print("lang_tokens", key, all_lang_data[key].shape)
    
    return (all_qpos_data, all_image_data, all_lang_data) 


def merge_act(actions_for_curr_step, k = 0.01):
    actions_populated = np.all(actions_for_curr_step != 0, axis=1)
    actions_for_curr_step = actions_for_curr_step[actions_populated]

    exp_weights = np.exp(-k * np.arange(actions_for_curr_step.shape[0]))
    exp_weights = (exp_weights / exp_weights.sum()).reshape((-1, 1))
    raw_action = (actions_for_curr_step * exp_weights).sum(axis=0)

    return raw_action

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

    parser.add_argument('--backbone', action='store', type=str, default='resnet34', help='visual backbone, choose from resnet18, resnet34, dino_v2, CLIP', required=False)
    parser.add_argument('--lang-backbone', action='store', type=str, default='CLIP', help='language backbone, choose from CLIP, onehot', required=False)
    parser.add_argument('--state_dim', action='store', type=int, default=7, help='state_dim', required=False)
    parser.add_argument('--action_dim', action='store', type=int, default=7, help='action_dim', required=False)
    
    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--save_jit', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--resumeid', action='store', default="", type=str, help='resume id', required=False)
    parser.add_argument('--resume_ckpt', action='store', default="", type=str, help='resume ckpt', required=False)
    parser.add_argument('--task-name', action='store', type=str, help='task name', required=True)
    parser.add_argument('--exptid', action='store', type=str, help='experiment id', required=True)
    parser.add_argument('--config-path', action='store', type=str, help='path_to_config_of_datasets', required=True)
    
    parser.add_argument('--saving-interval', action='store', type=int, default=10000, help='saving interval', required=False)
    args = vars(parser.parse_args())
    
    print(f"LOG dir: {LOG_DIR}\n")
    
    config = make_config(args)
    
    ckpt_dir = config['ckpt_dir']
    norm_stat_path = Path(ckpt_dir) / "dataset_stats.pkl"
    
    temporal_agg = True
    action_dim = args['action_dim']

    chunk_size = args['chunk_size']
    device = "cuda"
    
    timestamps = 220 # max length of an episode

    norm_stats = get_norm_stats(norm_stat_path)
    policy = load_policy(config)
    policy.cuda()
    policy.eval()

    history_stack = 0
    last_action_queue = None
    last_action_data = None
    
    # ---
    dataset_config = yaml.safe_load(open(args['config_path'], 'r'))
    dataset_paths = dataset_config['dataset_paths']

    model_name = "openai/clip-vit-base-patch32"
    cache_name = "/home/zhaoyixiu/ISR_project/CLIP/tokenizer"
    CLIP_tokenizer = CLIPTokenizer.from_pretrained(cache_name)

    num_tasks = len(dataset_paths)
    all_tasks_success_rates = []
    finished_idx = 0
    
    for task_idx in range(num_tasks):

        if task_idx < finished_idx:
            print("task", task_idx, "already finished. Move on to the next task.")
            continue
        
        print("start evaluating task ", task_idx + 1, "/", num_tasks)
        
        bsz = 10
        player = Player(dataset_paths[task_idx], num_envs = bsz, lang=dataset_config['lan_instructions'][task_idx])

        if temporal_agg:
            all_time_actions = torch.zeros([bsz, timestamps, timestamps+chunk_size, action_dim], device='cuda')
        else:
            num_actions_exe = chunk_size
        
        try:
            output = None
            act_index = 0
            
            num_episodes = 2
            success_count = 0
            video_out = []
            for episode_idx in range(num_episodes):
                print("Episode", episode_idx)
                
                for t in tqdm(range(timestamps)):
                    if history_stack > 0:
                        last_action_data = np.array(last_action_queue)

                    state, lang, images = player.get_state_and_images()
                    data = normalize_input(state, lang, images, norm_stats, CLIP_tokenizer, bsz=bsz)

                    if temporal_agg:
                        output = policy(*data).detach() #.cpu().numpy() # (bsz,chuck_size,action_dim)
                        
                        all_time_actions[:, t, t:t+chunk_size] = output
                
                        num_answers_t = min(t+1, chunk_size)
                        all_actions_at_t = all_time_actions[:, t-num_answers_t+1:t+1, t, :]
                        k = 0.01
                        weights = torch.exp(-k * torch.arange(num_answers_t).float()).to(device)
                        weights = weights / torch.sum(weights)
                        raw_action = torch.sum(all_actions_at_t * weights.unsqueeze(-1), dim=1)
                        
                        act = raw_action.cpu().numpy()
                        
                        # import pdb; pdb.set_trace()
                        
                        # all_time_actions[[t], t:t+chunk_size] = output
                        # act = merge_act(all_time_actions[:, t])
                    
                    # import ipdb; ipdb.set_trace()
                    
                    act = act * norm_stats["action_std"] + norm_stats["action_mean"]
                    reward, done = player.step(act)
                
                for k in range(bsz):
                    success_count += int(player.dones[k])
                
                video_out.extend(player.get_episode_recording())
                player.reset()
            
            print("Success rate:", success_count / (num_episodes * bsz))
            all_tasks_success_rates.append(success_count / (num_episodes * bsz))
            
            player.render_multiple_episode_video(video_out, os.path.join(current_dir, 'videos'), f"{config['task_name']}_{config['exptid']}_eval_{config['resume_ckpt']}_task_{task_idx}.mp4")
        
        except KeyboardInterrupt:
            player.end()
            exit()
        
        player.end()
    
    num_tasks = 0
    if 'generalize_dataset' in dataset_config:
        gen_dataset_path = dataset_config['generalize_dataset']
        num_tasks = len(gen_dataset_path)
    for task_idx in range(num_tasks):
        
        print("start evaluating task (unseen) ", task_idx + 1, "/", num_tasks)
        
        bsz = 10
        player = Player(gen_dataset_path[task_idx], num_envs = bsz, lang=None)

        if temporal_agg:
            all_time_actions = torch.zeros([bsz, timestamps, timestamps+chunk_size, action_dim], device='cuda')
        else:
            num_actions_exe = chunk_size
        
        try:
            output = None
            act_index = 0
            
            num_episodes = 2
            success_count = 0
            video_out = []
            for episode_idx in range(num_episodes):
                print("Episode", episode_idx)
                
                for t in tqdm(range(timestamps)):
                    if history_stack > 0:
                        last_action_data = np.array(last_action_queue)

                    state, lang, images = player.get_state_and_images()
                    data = normalize_input(state, lang, images, norm_stats, CLIP_tokenizer, bsz=bsz)

                    if temporal_agg:
                        output = policy(*data).detach() #.cpu().numpy() # (bsz,chuck_size,action_dim)
                        
                        all_time_actions[:, t, t:t+chunk_size] = output
                
                        num_answers_t = min(t+1, chunk_size)
                        all_actions_at_t = all_time_actions[:, t-num_answers_t+1:t+1, t, :]
                        k = 0.01
                        weights = torch.exp(-k * torch.arange(num_answers_t).float()).to(device)
                        weights = weights / torch.sum(weights)
                        raw_action = torch.sum(all_actions_at_t * weights.unsqueeze(-1), dim=1)
                        
                        act = raw_action.cpu().numpy()
                        
                        # import pdb; pdb.set_trace()
                        
                        # all_time_actions[[t], t:t+chunk_size] = output
                        # act = merge_act(all_time_actions[:, t])
                    
                    # import ipdb; ipdb.set_trace()
                    
                    act = act * norm_stats["action_std"] + norm_stats["action_mean"]
                    reward, done = player.step(act)
                
                for k in range(bsz):
                    success_count += int(player.dones[k])
                
                video_out.extend(player.get_episode_recording())
                player.reset()
            
            print("Success rate:", success_count / (num_episodes * bsz))
            all_tasks_success_rates.append(success_count / (num_episodes * bsz))
            
            player.render_multiple_episode_video(video_out, os.path.join(current_dir, 'videos'), f"{config['task_name']}_{config['exptid']}_eval_{config['resume_ckpt']}_generalize_task_{task_idx}.mp4")
        
        except KeyboardInterrupt:
            player.end()
            exit()
        
        player.end()
            
    print("success rate on all tasks:", all_tasks_success_rates)