import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import time 
from einops import rearrange
import IPython
e = IPython.embed
from pathlib import Path

import imageio
import torchvision
import json
import yaml

import init_path

from transformers import CLIPModel, CLIPTokenizer

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_paths, lan_instructions, camera_names, norm_stats, episode_len, history_stack=0):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.max_pad_len = 200

        self.history_stack = history_stack

        self.is_sims = []
        self.original_action_shapes = []

        self.states = []
        self.image_dict = dict()
        for cam_name in self.camera_names:
            self.image_dict[cam_name] = []
        self.actions = []
        self.language_instructions = []
        
        # model_name = "openai/clip-vit-base-patch32"
        self.CLIP_tokenizer = CLIPTokenizer.from_pretrained("/home/zhaoyixiu/ISR_project/CLIP/tokenizer")

        # Read in all the data
        for idx, (dataset_path, lan_ins) in enumerate(zip(dataset_paths, lan_instructions)):
            num_episodes = find_episodes_num(dataset_path)
            with h5py.File(dataset_path, 'r') as f:
                root = f['data']
                for episode_id in range(1, num_episodes+1):
                    self.is_sims.append(None)
                    self.original_action_shapes.append(root[f'demo_{episode_id}']['actions'][()].shape)
                    
                    ep_meta_dict = json.loads(root[f'demo_{episode_id}'].attrs['ep_meta'])
                    lan_ins_from_ep = ep_meta_dict['lang']

                    self.states.append(np.array(root[f'demo_{episode_id}']['obs']['robot0_joint_pos'][()]))
                    for cam_name in self.camera_names:
                        imgs = root[f'demo_{episode_id}']['obs'][cam_name][()]
                        # rearrange the image data to be in the form of (time, C, H, W)
                        imgs = rearrange(imgs, 't h w c -> t c h w')
                        self.image_dict[cam_name].append(imgs)
                    self.actions.append(np.array(root[f'demo_{episode_id}']['actions'][()]))
                    if lan_ins is not None:
                        self.language_instructions.append(lan_ins)
                    else:
                        self.language_instructions.append(lan_ins_from_ep)
                        
                    # print("shape of state", self.states[-1].shape) # 7
                    # print("shape of actions", self.actions[-1].shape) # 12
                    # print("shape of imgs", self.image_dict[self.camera_names[0]][-1].shape) # 3, 128, 128

        # shuffle the data according to episode_ids
        self.states = [self.states[i] for i in self.episode_ids]
        for cam_name in self.camera_names:
            self.image_dict[cam_name] = [self.image_dict[cam_name][i] for i in self.episode_ids]
        self.actions = [self.actions[i] for i in self.episode_ids]
        self.language_instructions = [self.language_instructions[i] for i in self.episode_ids]
        self.is_sims = [self.is_sims[i] for i in self.episode_ids]
        self.original_action_shapes = [self.original_action_shapes[i] for i in self.episode_ids]

        self.is_sim = None 

        self.episode_len = episode_len
        self.cumulative_len = np.cumsum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        return episode_index, start_ts
    
    def __getitem__(self, ts_index):
        sample_full_episode = False # hardcode

        index, start_ts = self._locate_transition(ts_index)

        original_action_shape = self.original_action_shapes[index]
        episode_len = original_action_shape[0]

        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)

        # get observation at start_ts only
        qpos = self.states[index][start_ts]

        if self.history_stack > 0:
            last_indices = np.maximum(0, np.arange(start_ts-self.history_stack, start_ts)).astype(int)
            last_action = self.actions[index][last_indices, :]

        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = self.image_dict[cam_name][index][start_ts]
            
        lang_ins = self.language_instructions[index]
        lang_tokens = self.CLIP_tokenizer(
            lang_ins, 
            padding='max_length', 
            truncation=True, 
            max_length=25,
            return_tensors="pt"
        )#['input_ids']
        # print('shape of lang_tokens', lang_tokens.shape, type(lang_tokens))

        # get all actions after and including start_ts
        all_time_action = self.actions[index][:]

        # pad the last action for max_pad_len times
        all_time_action_padded = np.zeros((self.max_pad_len+original_action_shape[0], original_action_shape[1]), dtype=np.float32) 
        all_time_action_padded[:episode_len] = all_time_action
        all_time_action_padded[episode_len:] = all_time_action[-1]
        
        padded_action = all_time_action_padded[start_ts:start_ts+self.max_pad_len] 
        real_len = episode_len - start_ts

        is_pad = np.zeros(self.max_pad_len)
        is_pad[real_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        if self.history_stack > 0:
            last_action_data = torch.from_numpy(last_action).float()

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        if self.history_stack > 0:
            last_action_data = (last_action_data - self.norm_stats['action_mean']) / self.norm_stats['action_std']
            qpos_data = torch.cat((qpos_data, last_action_data.flatten()))
        # print(f"qpos_data: {qpos_data.shape}, action_data: {action_data.shape}, image_data: {image_data.shape}, is_pad: {is_pad.shape}, lang_tokens: {type(lang_tokens)}")
        
        return image_data, qpos_data, lang_tokens, action_data, is_pad


def get_norm_stats(dataset_paths, num_episodes):
    
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []
    
    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, 'r') as f:
            root = f['data']
            num_data_episodes = len(list(root.keys()))
            for episode_idx in range(1, num_data_episodes+1):
                qpos = root[f'demo_{episode_idx}']['obs']['robot0_joint_pos'][()]
                action = root[f'demo_{episode_idx}']['actions'][()]
                # print("shape of qpos and action are:", qpos.shape, action.shape)
                
                all_qpos_data.append(torch.from_numpy(qpos))
                all_action_data.append(torch.from_numpy(action))
                all_episode_len.append(len(qpos))
    
    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True).float()  # (episode, timstep, action_dim)
    action_std = all_action_data.std(dim=0, keepdim=True).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True).float()
    qpos_std = all_qpos_data.std(dim=0, keepdim=True).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}
    print("stats: ", stats)

    return stats, all_episode_len

def find_episodes_num(dataset_path):
    """
    Output number of episodes in the dataset.
    """
    with h5py.File(dataset_path, 'r') as f:
        root = f['data']
        episodes_num = len(list(root.keys())) 
        # print("all_keys=", list(root.keys()))
    return episodes_num

def BatchSampler(batch_size, episode_len_l, sample_weights=None):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(dataset_paths, lan_instructions, camera_names, batch_size_train, batch_size_val=None):
    # print(f'\nData from: ')
    num_episodes = 0
    for idx, (dataset_path, lan_ins) in enumerate(zip(dataset_paths, lan_instructions)):
        # print(f'{dataset_path} with language instruction: {lan_ins}')
        num_episodes += find_episodes_num(dataset_path)
    
    # No validation set
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:]
    print(f"Train episodes: {len(train_indices)}")
    
    # obtain normalization stats for qpos and action
    norm_stats, all_episode_len = get_norm_stats(dataset_paths, num_episodes)

    train_episode_len_l = [all_episode_len[i] for i in train_indices]
    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_paths, lan_instructions, camera_names, norm_stats, train_episode_len_l)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=16, prefetch_factor=2)

    return train_dataloader, None, norm_stats, train_dataset.is_sim

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def parse_id(base_dir, prefix):
    base_path = Path(base_dir)
    # Ensure the base path exists and is a directory
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(f"The provided base directory does not exist or is not a directory: \n{base_path}")

    # Loop through all subdirectories of the base path
    for subfolder in base_path.iterdir():
        if subfolder.is_dir() and subfolder.name.startswith(prefix):
            return str(subfolder), subfolder.name
    
    # If no matching subfolder is found
    return None, None

def find_all_ckpt(base_dir, prefix="policy_epoch_"):
    base_path = Path(base_dir)
    # Ensure the base path exists and is a directory
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError("The provided base directory does not exist or is not a directory.")

    ckpt_files = []
    for file in base_path.iterdir():
        if file.is_file() and file.name.startswith(prefix):
            ckpt_files.append(file.name)
    # find latest ckpt
    ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split(prefix)[-1].split('_')[0]), reverse=True)
    epoch = int(ckpt_files[0].split(prefix)[-1].split('_')[0])
    return ckpt_files[0], epoch


def make_grid(images, nrow=8, padding=2, normalize=False, pad_value=0):
    """Make a grid of images. Make sure images is a 4D tensor in the shape of (B x C x H x W)) or a list of torch tensors."""
    grid_image = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value).permute(1, 2, 0)
    return grid_image

class KaedeVideoWriter():
    def __init__(self, video_path, save_video=False, video_name=None, fps=30, single_video=True):
        self.video_path = video_path
        self.save_video = save_video
        self.fps = fps
        self.image_buffer = {}
        self.single_video = single_video
        self.last_images = {}
        if video_name is None:
            self.video_name = "video.mp4"
        else:
            self.video_name = video_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save(self.video_name)

    def reset(self):
        if self.save_video:
            self.last_images = {}

    def append_image(self, image, idx=0):
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            if idx not in self.last_images:
                self.last_images[idx] = None
            self.image_buffer[idx].append(image[::-1])

    def append_vector_obs(self, images):
        if self.save_video:
            for i in range(len(images)):
                self.append_image(images[i], i)

    def save(self, video_name=None, flip=False, bgr=False):
        if video_name is None:
            video_name = self.video_name
        img_convention = 1
        color_convention = 1
        if flip:
            img_convention = -1
        if bgr:
            color_convention = -1
        if self.save_video:
            os.makedirs(self.video_path, exist_ok=True)
            if self.single_video:
                video_name = os.path.join(self.video_path, video_name)
                video_writer = imageio.get_writer(video_name, fps=self.fps)
                for idx in self.image_buffer.keys():
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im[::img_convention, :, ::color_convention])
                video_writer.close()
            else:
                for idx in self.image_buffer.keys():
                    video_name = os.path.join(self.video_path, f"{idx}.mp4")
                    video_writer = imageio.get_writer(video_name, fps=self.fps)
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im[::img_convention, :, ::color_convention])
                    video_writer.close()
            print(f"Saved videos to {video_name}.")


if __name__ == '__main__':
    # load_data("/home/yifengz/dataset_absjoint_salt_smallrange_100.hdf5", ["agentview_rgb"], 64)
    
    if False:
        dataset_paths = [
            '/home/zhaoyixiu/ISR_project/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams.hdf5',
            '/home/zhaoyixiu/ISR_project/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27/demo_gentex_im128_randcams.hdf5'
        ]
        camera_names = ['robot0_agentview_left_image']
        lan_instructions = [
        # 'Pick up item from counter and place it to cabinent',
            None,
            None,
        ]
    
    config_file_path = 'config/data2.yml'
    config = yaml.safe_load(open(config_file_path, 'r'))
    
    dataset_paths = config['dataset_paths']
    camera_names = config['camera_names']
    lan_instructions = config['lan_instructions']
    
    print(f"dataset_paths: {dataset_paths}")
    print(f"camera_names: {camera_names}")
    print(f"lan_instructions: {lan_instructions}")
    
    load_data(dataset_paths, lan_instructions, camera_names, 64)