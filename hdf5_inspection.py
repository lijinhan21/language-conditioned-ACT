import h5py
import numpy as np
import torch
import json

def print_dict(data, indent=0):
    indent_str = '  ' * indent
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            print(f"{indent_str}    {key}: {val.shape}")
        elif isinstance(val, torch.Tensor):
            print(f"{indent_str}    {key}: {val.shape}")
        elif isinstance(val, dict):
            print(f"{indent_str}    {key}:")
            print_dict(val, indent + 1)
        else:
            print(f"{indent_str}    {key}: {val}")

def print_attrs(name, obj, indent=0):
    """Print attributes of the object."""
    if isinstance(obj, h5py.Group) or isinstance(obj, h5py.Dataset):
        attrs = dict(obj.attrs)
        if attrs:
            indent_str = '  ' * indent
            print(f"{indent_str}Attributes of {name}:")
            for key, value in attrs.items():
                if isinstance(value, np.ndarray):
                    print(f"{indent_str}  {key}: {value.shape}")
                elif isinstance(value, torch.Tensor):
                    print(f"{indent_str}  {key}: {value.shape}")
                elif isinstance(value, str):
                    if value[0] != '{':
                        print(f"{indent_str}  {key}: {value[:50]} ...")
                    else:
                        d = json.loads(value)
                        print(f"{indent_str}  {key}: ")
                        print_dict(d, indent+1)
                else:
                    print(f"{indent_str}  {key}: {value}")




def print_hdf5_structure(name, obj, indent=0):
    """Recursively print the structure of the HDF5 file."""
    
    # if indent >= 3:
    #     return
    
    if ('demo_' in name) and ('demo_0' not in name):
        return
    
    indent_str = '  ' * indent
    print(f"{indent_str}{name} ({type(obj)})")

    # Print attributes if the object has any
    print_attrs(name, obj, indent)
    
    if isinstance(obj, h5py.Group):
        print(f"{indent_str}Datasets of {name}")
        # If it's a group, recurse into its sub-groups/datasets
        for key, value in obj.items():
            print_hdf5_structure(key, value, indent + 1)
    elif isinstance(obj, h5py.Dataset):
        # Handle case where the value is a numpy array or torch tensor
        data = obj[()]
        if isinstance(data, np.ndarray):
            print(f"{indent_str}  {data.shape}")
        elif isinstance(data, torch.Tensor):
            print(f"{indent_str}  {data.shape}")
        
        # If the data is a dictionary, output its keys and values recursively
        elif isinstance(data, dict):
            print(f"{indent_str}  Value is a dictionary:")
            print_dict(data)
        elif isinstance(data, str):
            # Otherwise, print a preview of the data
            print(f"{indent_str}  {data[:20]}")  # Show the first 5 elements for preview
        else:
            print(f"{indent_str}  {data}") 

def print_hdf5_file_structure(file_path):
    """Open the HDF5 file and print its structure."""
    with h5py.File(file_path, 'r') as f:
        for key, value in f.items():
            print_hdf5_structure(key, value)

# Example usage
file_path = '/home/zhaoyixiu/ISR_project/LIBERO/libero/datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5'  # Replace with the path to your HDF5 file
print_hdf5_file_structure(file_path)

"""
data (<class 'h5py._hl.group.Group'>)
Attributes of data:
  bddl_file_name: libero/libero/bddl_files/libero_goal/open_the_midd ...
  env_args:
      type: 1
      env_name: Libero_Tabletop_Manipulation
      problem_name: libero_tabletop_manipulation
      bddl_file: chiliocosm/bddl_files/libero_goal/open_the_middle_layer_of_the_drawer.bddl
      env_kwargs:
        robots: ['Panda']
        controller_configs:
          type: OSC_POSE
          input_max: 1
          input_min: -1
          output_max: [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
          output_min: [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]
          kp: 150
          damping_ratio: 1
          impedance_mode: fixed
          kp_limits: [0, 300]
          damping_ratio_limits: [0, 10]
          position_limits: None
          orientation_limits: None
          uncouple_pos_ori: True
          control_delta: True
          interpolation: None
          ramp_ratio: 0.2
        bddl_file_name: chiliocosm/bddl_files/libero_goal/open_the_middle_layer_of_the_drawer.bddl
        has_renderer: False
        has_offscreen_renderer: True
        ignore_done: True
        use_camera_obs: True
        camera_depths: False
        camera_names: ['robot0_eye_in_hand', 'agentview']
        reward_shaping: True
        control_freq: 20
        camera_heights: 128
        camera_widths: 128
        camera_segmentations: None
  env_name: Libero_Tabletop_Manipulation ...
  macros_image_convention: opengl ...
  num_demos: 50
  problem_info:
      problem_name: libero_tabletop_manipulation
      domain_name: robosuite
      language_instruction: open the middle drawer of the cabinet
  tag: libero-v1 ...
  total: 7027
Datasets of data
  demo_0 (<class 'h5py._hl.group.Group'>)
  Attributes of demo_0:
    init_state: (79,)
    model_file: <mujoco model="base">
  <compiler angle="radian" m ...
    num_samples: 138
  Datasets of demo_0
    actions (<class 'h5py._hl.dataset.Dataset'>)
      (138, 7)
    dones (<class 'h5py._hl.dataset.Dataset'>)
      (138,)
    obs (<class 'h5py._hl.group.Group'>)
    Datasets of obs
      agentview_rgb (<class 'h5py._hl.dataset.Dataset'>)
        (138, 128, 128, 3)
      ee_ori (<class 'h5py._hl.dataset.Dataset'>)
        (138, 3)
      ee_pos (<class 'h5py._hl.dataset.Dataset'>)
        (138, 3)
      ee_states (<class 'h5py._hl.dataset.Dataset'>)
        (138, 6)
      eye_in_hand_rgb (<class 'h5py._hl.dataset.Dataset'>)
        (138, 128, 128, 3)
      gripper_states (<class 'h5py._hl.dataset.Dataset'>)
        (138, 2)
      joint_states (<class 'h5py._hl.dataset.Dataset'>)
        (138, 7)
    rewards (<class 'h5py._hl.dataset.Dataset'>)
      (138,)
    robot_states (<class 'h5py._hl.dataset.Dataset'>)
      (138, 9)
    states (<class 'h5py._hl.dataset.Dataset'>)
      (138, 79)
"""
