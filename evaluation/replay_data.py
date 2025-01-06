import math
import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path
import h5py
from tqdm import tqdm
import time
import cv2
import json

import torch
from einops import rearrange

import init_path
from act.utils import KaedeVideoWriter, make_grid

import robosuite

from pathlib import Path

from libero.libero import get_libero_path, benchmark
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.algos import *
from libero.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset
from libero.lifelong.metric import (
    evaluate_loss,
    evaluate_success,
    raw_obs_to_tensor_obs,
)
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    NpEncoder,
    compute_flops,
)

from pathlib import Path
current_dir = Path(__file__).parent.resolve()

class Player:
    def __init__(self, dataset, num_envs=1, lang=None):

        bddl_folder = get_libero_path("bddl_files")
        
        data_desc = dataset.split('/')[-1]
        data_desc = data_desc.replace('_demo.hdf5', '')
        data_desc = data_desc.replace('_', ' ')
        
        benchmark_name = 'libero_goal'
        benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
        descriptions = [benchmark_instance.get_task(i).language for i in range(10)]
        print("descriptions=", descriptions)
        
        task_id = -1
        for i in range(len(descriptions)):
            if descriptions[i] == data_desc:
                task_id = i
                break
        print("task_id=", task_id)
        
        task = benchmark_instance.get_task(task_id)
        bddl_file = os.path.join(bddl_folder, task.problem_folder, task.bddl_file)
        
        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
        }

        self.env_num = num_envs
        # self.env =  OffScreenRenderEnv(**env_args)
        self.env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(self.env_num)]
        )

        if lang is not None:
            self.lang = lang
        else:
            self.lang = data_desc
        self.task = task
        
        self.init_states = benchmark_instance.get_task_init_states(task_id)
        self.num_episodes = 0
        
        self.reset()        
        
        # import pdb; pdb.set_trace()
        
    def get_state_and_images(self):
        
        joint_states = []
        for i in range(self.env_num):
            joint_states.append(self.obs[i]['robot0_joint_pos'])
        
        agentview_rgb = []
        for i in range(self.env_num):
            agentview_rgb.append(self.obs[i]['agentview_image'])
            
        eye_in_hand_rgb = []
        for i in range(self.env_num):
            eye_in_hand_rgb.append(self.obs[i]['robot0_eye_in_hand_image'])
            
        images = []
        for i in range(self.env_num):
            images.append([agentview_rgb[i], eye_in_hand_rgb[i]])
        
        return joint_states, self.lang, images
    
    def step(self, action):
        
        self.obs, reward, done, info = self.env.step(action)
        
        cam_name = 'agentview_image'
        for i in range(self.env_num):
            frame = self.obs[i][cam_name]
            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if self.dones[i] == False:
                if done[i]:
                    print("Episode done!")
                    self.dones[i] = True
                    if reward[i] > 0.5:
                        green_filter = np.zeros_like(frame)
                        green_filter[:, :, 1] = 255
                        alpha = 0.2
                        success_img = cv2.addWeighted(frame, 1 - alpha, green_filter, alpha, 0)
                        self.episode_recording[i].append(success_img)
                    else:
                        red_filter = np.zeros_like(frame)
                        red_filter[:, :, 2] = 255
                        alpha = 0.2
                        fail_img = cv2.addWeighted(frame, 1 - alpha, red_filter, alpha, 0)
                        self.episode_recording[i].append(fail_img)
                else:
                    self.episode_recording[i].append(frame)
        
        time.sleep(1/20)
        
        return reward, done

    def end(self):
        self.env.close()
        del self.env
    
    def reset(self):
        
        self.env.reset()
        
        # indice = self.num_episodes % self.init_states.shape[0]
        indices = (self.num_episodes + np.arange(self.env_num)) % self.init_states.shape[0]
        self.obs = self.env.set_init_state(self.init_states[indices])
        self.num_episodes += self.env_num
        
        for _ in range(5):
            self.obs, _, _, _ = self.env.step(np.zeros((self.env_num, 7)))
        
        self.episode_recording = []
        self.dones = []
        for i in range(self.env_num):
            self.episode_recording.append([])
            self.dones.append(False)
        
        time.sleep(3)
        print("start new episode!", self.lang)
        
        
        
    def get_episode_recording(self):
        return self.episode_recording
    
    def render_single_episode_video(self, output_path, video_name):
        frames = self.get_episode_recording()
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        filename = Path(output_path) / f"{video_name}"
        os.makedirs(output_path, exist_ok=True)
        out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print("episode video saved at", filename)

    def render_multiple_episode_video(self, video_out, output_path, video_name):
        video_length = max([len(episode) for episode in video_out])
        video_out = [episode + [episode[-1]] * (video_length - len(episode)) for episode in video_out]
        
        print("Video_length=", video_length, "Making video...")
        video_writer = KaedeVideoWriter(output_path, save_video=True, video_name=video_name, fps=30, single_video=True)
        for i in range(video_length):
            images = []
            for episode in video_out:
                img = episode[i]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.flip(img, 0)
                images.append(torch.tensor(img))
            # images = [torch.tensor(episode[i]) for episode in video_out]
            for j in range(len(images)):
                images[j] = rearrange(images[j], 'h w c -> c h w')
            grid_image = make_grid(images, nrow=4).numpy()
            video_writer.append_image(grid_image)
        video_writer.save()

if __name__ == '__main__':
    
    pass
