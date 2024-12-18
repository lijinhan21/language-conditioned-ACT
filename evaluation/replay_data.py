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
from robocasa.utils.dataset_registry import (
    get_ds_path,
    SINGLE_STAGE_TASK_DATASETS,
    MULTI_STAGE_TASK_DATASETS,
)

from pathlib import Path
current_dir = Path(__file__).parent.resolve()

def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
        ep_meta_dict = json.loads(f["data"]["demo_1"].attrs['ep_meta'])
        env_meta['lang'] = ep_meta_dict['lang']
    else:
        raise ValueError
    f.close()
    return env_meta

class Player:
    def __init__(self, dataset):
        
        env_meta = get_env_metadata_from_dataset(dataset_path=dataset)
        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = True
        env_kwargs["use_camera_obs"] = False

        self.env = robosuite.make(**env_kwargs)
        
        self.reset()
        
        # import pdb; pdb.set_trace()
        
    def get_state_and_images(self):
        
        cos_joint = self.obs['robot0_joint_pos_cos']
        sin_joint = self.obs['robot0_joint_pos_sin']
        joint_states = np.arctan2(sin_joint, cos_joint)
        
        cam_name = 'robot0_agentview_left'
        rgb_img = self.env.sim.render(height=128, width=128, camera_name=cam_name)[::-1]
        
        return joint_states, self.lang, rgb_img
    
    def step(self, action):
        
        self.obs, reward, done, info = self.env.step(action)
        
        cam_name = 'robot0_agentview_left'
        frame = self.env.sim.render(height=512, width=512, camera_name=cam_name)[::-1]
        # frame = cv2.flip(frame, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.episode_recording.append(frame)
        
        if done:
            print("Episode done!")
            if reward > 0.5:
                green_filter = np.zeros_like(frame)
                green_filter[:, :, 1] = 255
                alpha = 0.2
                success_img = cv2.addWeighted(frame, 1 - alpha, green_filter, alpha, 0)
                self.episode_recording.append(success_img)
            else:
                red_filter = np.zeros_like(frame)
                red_filter[:, :, 2] = 255
                alpha = 0.2
                fail_img = cv2.addWeighted(frame, 1 - alpha, red_filter, alpha, 0)
                self.episode_recording.append(fail_img)
        
        time.sleep(1/20)
        
        return reward, done

    def end(self):
        del self.env
    
    def reset(self):
        self.obs = self.env.reset()
        self.episode_recording = []
        
        ep_meta = self.env.get_ep_meta()
        self.lang = ep_meta.get("lang", None)
        
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
    
    dataset_path = '/data/william/dataset_absjoint_ice_40.hdf5' # TODO: change with actual dataset path
    
    with h5py.File(dataset_path, 'r') as f:
        root = f['data']['demo_0']
        actions = np.array(root['actions'][()])
        agentview_rgb = np.array(root['obs']['agentview_rgb'][()])
        states = np.array(root['obs']['joint_states'][()])
    
    timestamps = states.shape[0]
    single_arm = (len(actions[0]) == 13)
    
    player = Player(single_arm)
    
    try:
        for t in tqdm(range(timestamps)):
            player.step(actions[t], agentview_rgb[t])
        player.render_single_episode_video(current_dir, 'test_dataset.mp4')
    except KeyboardInterrupt:
        player.end()
        exit()
