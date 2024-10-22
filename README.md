# UL-VIO (ECCV 2024)

<a href="https://arxiv.org/abs/2409.13106"><img src="https://img.shields.io/badge/Paper-arXiv:2409.13106-Green"></a>
[![project_page](https://img.shields.io/badge/-project%20page-skyblue)](https://jp4327.github.io/ulvio/) 

<!-- <a href="https://arxiv.org/abs/2409.13106"><img src="https://img.shields.io/badge/Paper-arXiv:2409.13106-Green"></a> -->


This repository contains the evaluation code for the following paper:

[**UL-VIO: Ultra-lightweight Visual-Inertial Odometry with Noise Robust Test-time Adaptation**](https://arxiv.org/abs/2409.13106)

<img src="figures/figure.png" alt="overview" width="700"/> 

>Abstract - Data-driven visual-inertial odometry (VIO) has received highlights for its performance since VIOs are a crucial compartment in autonomous robots.
However, their deployment on resource-constrained devices is non-trivial since large network parameters should be accommodated in the device memory.
Furthermore, these networks may risk failure post-deployment due to environmental distribution shifts at test time.
In light of this, we propose **UL-VIO** -- an ultra-lightweight (<1M) VIO network capable of test-time adaptation (TTA) based on visual-inertial consistency.
Specifically, we perform model compression to the network while preserving the low-level encoder part, including all BatchNorm parameters for resource-efficient test-time adaptation.
It achieves $36 \times$ smaller network size than state-of-the-art with a minute increase in error -- $1$\% on the KITTI dataset.
For test-time adaptation, we propose to use the inertia-referred network outputs as pseudo labels and update the BatchNorm parameter for lightweight yet effective adaptation.
To the best of our knowledge, this is the first work to perform noise-robust TTA on VIO.
Experimental results on the KITTI, EuRoC, and Marulan datasets demonstrate the effectiveness of our resource-efficient adaptation method under diverse TTA scenarios with dynamic domain shifts.

## Installation

      $git clone https://github.com/jp4327/ulvio
      $conda env create --file environment.yaml
      $conda activate ulvio

## Data Preparation

The code in this repository is tested on [KITTI](https://www.cvlibs.net/datasets/kitti/) Odometry dataset. The IMU data after pre-processing is provided under `data/imus`. To download the images and poses, please run

      $cd data
      $source data_prep.sh 

After running the script,`data` folder shall look like the following:
```
data
├── data_prep.sh
├── imus
│   ├── 00.mat
│   ├── 01.mat
│   ...
│   └── 10.mat
├── poses
│   ├── 00.txt
│   ├── 01.txt
│   ...
│   └── 10.txt
└── sequences
    ├── 00
    │   ├── calib.txt
    │   ├── image_2
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    |   |   | ...
    │   ├── image_3
    │   └── times.txt
    ├── 01
    ...
    └── 10
```

## Evaluation
Run pre-trained model with the following command:

      $python test.py

The resulting CLI output looks like:
```
Seq: 05, t_rel: 4.2233, r_rel: 1.3760, t_rmse: 0.0564, r_rmse: 0.0744, 
Seq: 07, t_rel: 3.2585, r_rel: 2.6176, t_rmse: 0.0594, r_rmse: 0.0873, 
Seq: 10, t_rel: 3.0715, r_rel: 1.2113, t_rmse: 0.0611, r_rmse: 0.0859, 
```

## Acknowledgements

This code is borrowed heavily from [NASVIO](https://github.com/unchenyu/NASVIO) and [Visual-Selective-VIO](https://github.com/mingyuyng/Visual-Selective-VIO). We thank the authors for sharing their codes.

## Reference

If you find our work useful in your research, please consider citing our paper:

> Jinho Park, Se Young Chun, Mingoo Seok, "UL-VIO: Ultra-lightweight Visual-Inertial Odometry with Noise Robust Test-time Adaptation"

      @article{park2024ulvio,
            author    = {Park, Jinho 
                        and Chun, Se Young 
                        and Seok, Mingoo},
            title     = {UL-VIO: Ultra-lightweight Visual-Inertial Odometry with Noise Robust Test-time Adaptation},
            journal   = {ECCV},
            year      = {2024},
      }
