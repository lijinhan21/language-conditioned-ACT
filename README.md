
# ACT

## Dev:

Training:

```
python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name test --exptid 01-test --config-path config/data2.yml

python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name test --exptid 01-test --config-path config/data1.yml

python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 2000000 --lr 5e-5 --seed 0 --task-name data1 --exptid 3000mg-long --config-path config/data1.yml     

python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name libero1 --exptid open-middle-drawer --config-path config/data_libero_1.yml

python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3 --exptid first-three-tasks --config-path config/data_libero_3.yml

[running: onehot, 3 tasks] python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3-revise --exptid first-three-tasks-onehot --config-path config/data_libero_3.yml --lang-backbone OneHot

[running: CLIP, 10 tasks]
CUDA_VISIBLE_DEVICES=1 python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 250000 --lr 5e-5 --seed 0 --task-name libero3 --exptid ten-tasks --config-path config/data_libero_10.yml
```

Evaluation:

```
python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name test --exptid 01-test --config-path config/data2.yml --resume_ckpt 40000

python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name data1 --exptid 3000mg --config-path config/data1.yml --resume_ckpt 40000

python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name libero1 --exptid open-middle-drawer --config-path config/data_libero_1.yml --resume_ckpt 10000

CUDA_VISIBLE_DEVICES=2 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3 --exptid first-three-tasks --config-path config/data_libero_3.yml --resume_ckpt 40000

CUDA_VISIBLE_DEVICES=2 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3-revise --exptid first-three-tasks-onehot --config-path config/data_libero_3.yml --lang-backbone OneHot --resume_ckpt 10000
```

## Installation

```
conda create -n aloha python=3.8.10
conda activate aloha
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install dm_control==1.0.14
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython
pip install wandb
pip install imageio
pip install transformers
cd detr && pip install -e .
```

For simulation evaluation, you also need to:
```
git clone git@github.com:lijinhan21/sim_OKAMI.git
cd sim_OKAMI
pip install -r requirements.txt
pip install -r requirements_okami.txt
```

## Training Guide

1. You can verify the image and action sequences of a specific episode in the dataset using ``replay_data.py``. The results will be saved in `test_dataset.mp4`.

2. To train ACT, run:
```
    python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name test --exptid 01-test --dataset-path /home/yifengz/dataset_absjoint_salt_smallrange_100.hdf5
```
<!-- 
After training, you can save jit for the desired checkpoint: 
```
    python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name test --exptid 01-test --dataset-path /home/yifengz/dataset_absjoint_salt_smallrange_100.hdf5 --save_jit --resume_ckpt 25000
``` -->


3. You can visualize the trained policy with inputs from dataset using ``replay_policy.py``, example usage: (basically, just add `--resume ckpt xxx` behind the training args, and change the script name)
```
    python replay_policy.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name test --exptid 01-test --dataset-path /home/yifengz/dataset_absjoint_salt_smallrange_100.hdf5 --resume_ckpt 200
```
The results will be saved in `{task-name}_{exptid}_replay.mp4`.

If you are using simulation data, you can also evaluate the policy in simulation:
```
    python sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name test --exptid 01-test --dataset-path /home/yifengz/dataset_absjoint_salt_smallrange_100.hdf5 --resume_ckpt 200
```
The results will be saved in `{task-name}_{exptid}_eval.mp4`.

---

This part of the codebase is modified from OpenTelevision.
```
@article{cheng2024tv,
title={Open-TeleVision: Teleoperation with Immersive Active Visual Feedback},
author={Cheng, Xuxin and Li, Jialong and Yang, Shiqi and Yang, Ge and Wang, Xiaolong},
journal={arXiv preprint arXiv:2407.01512},
year={2024}
}
```