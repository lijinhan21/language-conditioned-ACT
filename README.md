
# ACT for OKAMI

## Training Guide

1. Process the specified dataset for training using ``post_process.py``.

2. You can verify the image and action sequences of a specific episode in the dataset using ``replay_demo.py``.

3. To train ACT, run:
```
    python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --taskid 00 --exptid 01-sample-expt
```

4. After training, save jit for the desired checkpoint:
```
    python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --taskid 00 --exptid 01-sample-expt\
                               --save_jit --resume_ckpt 25000
```

5. You can visualize the trained policy with inputs from dataset using ``deploy_sim.py``, example usage:
```
    python deploy_sim.py --taskid 00 --exptid 01 --resume_ckpt 25000
```

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