
# ACT

## Dev:

Training:

```
python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name test --exptid 01-test --config-path config/data2.yml

python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name test --exptid 01-test --config-path config/data1.yml

python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 2000000 --lr 5e-5 --seed 0 --task-name data1 --exptid 3000mg-long --config-path config/data1.yml     

python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name libero1 --exptid open-middle-drawer --config-path config/data_libero_1.yml

python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3 --exptid first-three-tasks --config-path config/data_libero_3.yml

python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3-revise --exptid first-three-tasks-onehot --config-path config/data_libero_3.yml --lang-backbone OneHot

CUDA_VISIBLE_DEVICES=1 python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 250000 --lr 5e-5 --seed 0 --task-name libero3 --exptid ten-tasks --config-path config/data_libero_10.yml

CUDA_VISIBLE_DEVICES=0 python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name libero1 --exptid T4 --config-path config/data_libero_1_T4.yml

CUDA_VISIBLE_DEVICES=0 python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 40003 --lr 5e-5 --seed 0 --task-name libero1 --exptid T8 --config-path config/data_libero_1_T8.yml

CUDA_VISIBLE_DEVICES=1 python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3 --exptid 3tasks-onehot --config-path config/data_libero_3.yml --lang-backbone OneHot

[revise one-hot embedding. Worked!!!!!!!] CUDA_VISIBLE_DEVICES=0 python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3 --exptid 3tasks-onehot-revise --config-path config/data_libero_3.yml --lang-backbone OneHot

[onehot in lang ins] CUDA_VISIBLE_DEVICES=2 python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3 --exptid 3tasks-onehot-lang --config-path config/data_libero_3_onehot.yml

[T459: finished] CUDA_VISIBLE_DEVICES=0 python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3 --exptid T459 --config-path config/data_libero_3_T459.yml 

[T459_onehot: finished] CUDA_VISIBLE_DEVICES=0 python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3 --exptid onehot-T459 --config-path config/data_libero_3_T459.yml --lang-backbone OneHot

[9tasks_CLIP: finished] CUDA_VISIBLE_DEVICES=1 python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 250005 --lr 5e-5 --seed 0 --task-name libero3 --exptid 9tasks-CLIP --config-path config/data_libero_9.yml 

[9tasks_onehot: finished] CUDA_VISIBLE_DEVICES=0 python act/imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 250005 --lr 5e-5 --seed 0 --task-name libero3 --exptid 9tasks-onehot --config-path config/data_libero_9.yml --lang-backbone OneHot
```

Evaluation:

```
python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name test --exptid 01-test --config-path config/data2.yml --resume_ckpt 40000

python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name data1 --exptid 3000mg --config-path config/data1.yml --resume_ckpt 40000

CUDA_VISIBLE_DEVICES=1 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name libero1 --exptid open-middle-drawer --config-path config/data_libero_1.yml --resume_ckpt 40000

CUDA_VISIBLE_DEVICES=1 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3 --exptid first-three-tasks --config-path config/data_libero_3.yml --resume_ckpt 70000

CUDA_VISIBLE_DEVICES=2 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3-revise --exptid first-three-tasks-onehot --config-path config/data_libero_3.yml --lang-backbone OneHot --resume_ckpt 10000


CUDA_VISIBLE_DEVICES=1 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3-revise --exptid first-three-tasks-onehot --config-path config/data_libero_3.yml --lang-backbone OneHot --resume_ckpt 70000

CUDA_VISIBLE_DEVICES=1 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 250000 --lr 5e-5 --seed 0 --task-name libero3 --exptid ten-tasks --config-path config/data_libero_10.yml --resume_ckpt 240000

CUDA_VISIBLE_DEVICES=0 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --task-name libero1 --exptid T4 --config-path config/data_libero_1_T4.yml --resume_ckpt 40000

CUDA_VISIBLE_DEVICES=1 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3 --exptid 3tasks-onehot --config-path config/data_libero_3.yml --lang-backbone OneHot --resume_ckpt 70000

CUDA_VISIBLE_DEVICES=2 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 40003 --lr 5e-5 --seed 0 --task-name libero1 --exptid T8 --config-path config/data_libero_1_T8.yml --resume_ckpt 40000


CUDA_VISIBLE_DEVICES=1 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3 --exptid 3tasks-onehot-revise --config-path config/data_libero_3.yml --lang-backbone OneHot --resume_ckpt 60000

CUDA_VISIBLE_DEVICES=2 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3 --exptid 3tasks-onehot-lang --config-path config/data_libero_3_onehot.yml --resume_ckpt 40000

[T459 one-hot] CUDA_VISIBLE_DEVICES=0 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3 --exptid onehot-T459 --config-path config/data_libero_3_T459.yml --lang-backbone OneHot --resume_ckpt 70000

[T459 CLIP] CUDA_VISIBLE_DEVICES=1 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 80000 --lr 5e-5 --seed 0 --task-name libero3 --exptid T459 --config-path config/data_libero_3_T459.yml --resume_ckpt 70000

[9tasks_CLIP] CUDA_VISIBLE_DEVICES=1 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 250005 --lr 5e-5 --seed 0 --task-name libero3 --exptid 9tasks-CLIP --config-path config/data_libero_9.yml --resume_ckpt 250000

[9tasks_onehot] CUDA_VISIBLE_DEVICES=2 python evaluation/sim_evaluation.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 250005 --lr 5e-5 --seed 0 --task-name libero3 --exptid 9tasks-onehot --config-path config/data_libero_9.yml --lang-backbone OneHot --resume_ckpt 250000
```

### TODOs

- [x] Finish training all multi-task policies (4 in total).
    - [x] CLIP 3 tasks. (80000 epochs)
    - [x] CLIP 10 tasks. (250000 epochs)
    - [x] OneHot 3 tasks.
    - [x] CLIP T459 
    - [x] OneHot T459 
    - [x] CLIP 9 tasks.
    - [x] OneHot 9 tasks.

- [x] Implement parallel evaluation (multiple environments at a time).

- [x] Finish the evaluation of all multi-task policies.
    - [x] CLIP 3 tasks. 
    - [x] CLIP 10 tasks.
    - [x] OneHot 3 tasks.
    - [x] CLIP T459
    - [x] OneHot T459 
    - [x] CLIP 9 tasks
    - [x] OneHot 9 tasks

- [x] Debug OneHot embedding.
    - [x] Add mlp to calculate embedding. (Won't work)
    - [x] Add more regularization in mlp. (Worked!!!!!!!!!!!)
    - [x] Use language index. (Running)

- [x] Evaluate on the task "put the wine bottle on top of the cabinet".

- [ ] Finish training selective single-task policy.
    - [x] T1 CLIP
    - [x] T4 CLIP
    - [x] T8 CLIP
    - [ ] T1 OneHot
    - [ ] T4 OneHot
    - [ ] T8 OneHot

### Experiment Plan:

**Comparison between CLIP embedding and one-hot vectors on seen tasks**

Single-task policy:

| | T1 | T4 | T8 |
| ----- | ----- | ----- | ----- |
|CLIP| 80% (20000epochs) | 75% (40000epochs)| 85% (40000 epochs)|
|OneHot| | | |
<!-- |CLIP-randemb| | | |
|OneHot-randemb| | | | -->

3 tasks (data_libero_3.yml) :

Evaluated over 20 trials. Trained 70000 epochs.

|  | T1 | T2 | T3 |
| ----- | ----- | ----- | ----- |
| CLIP | 95% | 60% | 80% |
| OneHot(MLP) | 40% | 45% | 0% |
| OneHot(MLP+initialization) | 80% | 80% | 55% |
<!-- | CLIP-randemb |  |  |  |
| OneHot-randemb |  |  | | -->

3 tasks new. (data_libero_3_T459.yml)

Note: this is for testing generalization to new task.

T4 : put the bowl on the plate
T9 : put the wine bottle on top of the cabinet
T5 : put the bowl on the stove

for testing:
T6 : put the bowl on top of the cabinet

|  | T4 | T5 | T9 |
| ----- | ----- | ----- | ----- |
| CLIP | 80% | 80% | 100% |
| OneHot | 80% | 95% | 90% |

9 tasks (data_libero_9.yml)

| | T1 | T2 | T3 | T4 | T5 | T7 | T8 | T9 | T10 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| CLIP (25000) | 75% | 60% | 70% | 55% | 80% | 75% | 65% | 75% | 100% |
| CLIP (24000) | 95% | 70% | 55% | 50% | 90% | 70% | 75% | 75% | 100% |
| OneHot (25000) | 95% | 60% | 55% | 70% | 75% | 60% | 75% | 50% | 100% |
| OneHot (24000) | 100% | 75% | 55% | 70% | 80% | 60% | 80% | 65% | 90% |


10 tasks (data_libero_10.yml):

Evaluated over 20 trials. Trained 240000 epochs.

| | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| CLIP | 85% | 75% | 65% | 60% | 75% | 95% | 60% | 70% | 70% | 95% |
<!-- | OneHot | | | | | | | | | | | -->
<!-- | CLIP-randemb | | | | | | | | | | |
| OneHot-randemb | | | | | | | | | | | -->

**Test CLIP and one-hot on unseen task**

| | CLIP-3 | CLIP-9  | OneHot-3 | OneHot-9 |
| ----- | ----- | ----- | ----- | ----- |
|T6| 0% | 5% | 0% | 0% |

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