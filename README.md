# MANSY_ImmersiveVideoStreaming
Implementation of paper "[MANSY: Generalizing Neural Adaptive Immersive Video Streaming With Ensemble and Representation Learning](https://arxiv.org/abs/2311.06812)" and a trace-driven simulator for immsersive (or, omnidirectional, 360-degree) video streaming.

MANSY is short for "ense**M**ble and represent**A**tion lear**N**ing based **SY**stem for tile-based immersive video streaming". MANSY is also named after Duo's beloved, as a way of expressing gratitude for her generous support towards Duo's research career.

Paper:
<!-- > The popularity of immersive videos has prompted extensive research into neural adaptive tile-based streaming to optimize video transmission over networks with limited bandwidth. However, the diversity of users' viewing patterns and Quality of Experience (QoE) preferences has not been fully addressed yet by existing neural adaptive approaches for viewport prediction and bitrate selection. Their performance can significantly deteriorate when users' actual viewing patterns and QoE preferences differ considerably from those observed during the training phase, resulting in poor generalization. In this paper, we propose MANSY, a novel streaming system that embraces user diversity to improve generalization. Specifically, to accommodate users' diverse viewing patterns, we design a Transformer-based viewport prediction model with an efficient multi-viewport trajectory input output architecture based on implicit ensemble learning. Besides, we for the first time combine the advanced representation learning and deep reinforcement learning to train the bitrate selection model to maximize diverse QoE objectives, enabling the model to generalize across users with diverse preferences. Extensive experiments demonstrate that MANSY outperforms state-of-the-art approaches in viewport prediction accuracy and QoE improvement on both trained and unseen viewing patterns and QoE preferences, achieving better generalization. -->
```
@article{duo2023mansy,
  title={MANSY: Generalizing Neural Adaptive Immersive Video Streaming With Ensemble and Representation Learning},
  author={Wu, Duo and Wu, Panlong and Zhang, Miao and Wang, Fangxin},
  journal={arXiv preprint arXiv:2311.06812},
  year={2023}
}
```

## Requirements
```
python>=3.8
torch<=2.0
tianshou== 0.4.8
prettytable==3.5.0
numpy==1.24.3
gym==0.26.2
ffmpeg==4.2.2
```
Note: Other environment settings may work, but we didn't test on them.

## Folder Content
- `dataset_preprocess`: This folder contains scripts for preprocessing datasets, such as extracting and simplifying viewport files, extracting video chunk information.
- `viewport_prediction`: This folder contains codes for immsersive video viewport prediction.
- `bitrate_selection`: This folder contains codes for immsersive video bitrate selection.
- `datasets`: This folder contains datasets.
- `models`: This folder contains model checkpoints saved during training. Examples are provided in this folder to better understand the file structure.
- `results`: This folder contains training/testing results files. Examples are provided in this folder to better understand the file structure.

## Uage
### Part 1: Preprocess Dataset
#### Step 1: Download Video and Viewport Datasets

Frist, we need to download datasets and save them in folder `datasets`. For example, let's say we want to download Jin2022 dataset used in our paper. Download it from here: https://cuhksz-inml.github.io/head_gaze_dataset/.

Save the videos in the dataset in the folder `datasets/raw/Jin2022/videos`, and please organize the videos in the following format:
```
datasets/raw/Jin2022/videos/
    video1/  # video{id}
        1-1M.mp4  # {id}-{bitrate_version}.mp4
        1-5M.mp4
        1-8M.mp4
        1-16M.mp4
        1-35M.mp4
    video2/
        ...
```
Save the viewports files in the dataset in the folder `datasets/raw/Jin2022/viewports`, and please organize the files in the following format:
```
datasets/raw/Jin2022/videos/
    1/  # {user_id}
        file1.csv  
        file2.csv 
        file3.csv 
        file4.csv 
        file5.csv 
    2/
        ...
```
Note that if you determine to use the Jin2022 dataset, please kindly cite the following paper:
```
@inproceedings{jin2022you,
  title={Where Are You Looking? A Large-Scale Dataset of Head and Gaze Behavior for 360-Degree Videos and a Pilot Study},
  author={Jin, Yili and Liu, Junhua and Wang, Fangxin and Cui, Shuguang},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={1025--1034},
  year={2022}
}
```
Besides, the original Jin2022 dataset does not contain videos in different bitrate versions, so you need to manually transform them (possibly using ffmpeg).

#### Step 2: Download Bandwidth Dataset 

If you want to use the bitrate selection codes, you will also need to download a bandwidth dataset for simulating network conditions. Let's say we want to download `4G` dataset used in our paper. Download it from the following paper:
```
@article{van2016http,
  title={HTTP/2-based adaptive streaming of HEVC video over 4G/LTE networks},
  author={Van Der Hooft, Jeroen and Petrangeli, Stefano and Wauters, Tim and Huysegems, Rafael and Alface, Patrice Rondao and Bostoen, Tom and De Turck, Filip},
  journal={IEEE Communications Letters},
  volume={20},
  number={11},
  pages={2177--2180},
  year={2016},
  publisher={IEEE}
}
```
Save the dataset in the folder `datasets/raw_network/4G`, and please organize the files in the following formats:
```
datasets/raw_network/4G/
    file1
    file2
    file3
    file4
    file5
```
#### Step 3: Process Datasets
Once you complete download and file organization, you can use the scripts in `dataset_preprocess` to process the datasets.

First, change the working directory into `dataset_preprocess`:
```sh
cd dataset_preprocess
```

Run the following command to process viewports/videos/bandwidths datasets:
```sh
python hmdtrace.py  --dataset Jin2022  # process viewports
python video.py --dataset Jin2022  # process videos
python network.py --dataset 4G  # process bandwidths
```

The processed files will be saved at:
```
datasets/Jin2022/viewports/ 
datasets/Jin2022/video_manifests/
datasets/network/4G/
```

#### Add New Datasets
If you want to add a new dataset, in addition to following the above steps, you also need to change the following codes/files:
- `config.yml`: Please add the information of the new dataset here. 
- `dataset_preprocess/hmdtrace.py`: Please specify the way to process the new viewport dataset here.
- `dataset_preprocess/network.py`: Please specify the way to process the new network dataset here.

### Part 2: Viewport Prediction
If you want to play viewport prediction, first change the working directory:
```sh
cd viewport_prediction
```

By simply running the `run_model.py` with some arguments, you can run a viewport prediction model. For example:
```
python run_models.py --model mtio --train --test --train-dataset Jin2022 --test-dataset Jin2022 --his-window 5 --fut-window 15 --bs 512 --seed 5 --dataset-frequency 5 --sample-step 5 --hidden-dim 512 --block-num 2 --lr 0.0001 --epochs 200 --epochs-per-valid 3 --device cuda:0
```
Check file `run_model.py` for the detailed explanation of the command arguments.

You can customize your own models following the example of `run_model.py`.

### Part 3: Bitrate Selection
If you want to play bitrate selection, first change the working directory to `viewport_prediction`:
```sh
cd viewport_prediction
```

Run the `predict.py` to generate files of predicted viewports for the bitrate selection experiments:
```sh
python predict.py --model regression --device cpu --dataset Jin2022 --bs 64 --seed 1
```
You can use other models (with path to the model checkpoints) to generate predicted viewports. Note that this generation can be done once for a lifetime.

Next, change the working directory to `bitrate_selection`:
```sh
cd ..
cd bitrate_selection
```

By simply running the `run_simple_rl.py` with some arguments, you can run a bitrate selection model. For example:
```sh
python run_simple_rl.py --epochs 100 --step-per-epoch 6000 --step-per-collect 2000 --batch-size 256 --train --train-dataset Jin2022 --test --test-dataset Jin2022 --qoe-train-id 0 --qoe-test-ids 0 --test-on-seen --device cuda:0 --seed 1
```
You can customize your own models following the example of `run_simple_rl.py`.

If you want to try *MANSY*, try the following command:
```sh
python run_mansy.py --train --test --epoch 1000 --step-per-epoch 4096 --step-per-collect 4096 --lr 0.0005 --batch-size 512 --train --train-dataset Jin2022 --test --test-dataset Jin2022 --qoe-test-ids 0 1 2 3 --test-on-seen --lamb 0.5 --train-identifier --identifier-epoch 1000 --identifier-lr 0.0001 --device cuda:0 --gamma 0.95 --ent-coef 0.02 --seed 5 --use-identifier
```
Check file `run_simple_rl.py/run_mansy.py` for the detailed explanation of the command arguments.

#### Behavior Cloning (BC) Initialization
Behavior cloning (BC) is a common trick to initialize the deep inforcement learning (DRL) model. Our codes also support BC initialization.

In our codes, we use an MPC-based policy with perfect knowledge of the trainin environment as the expert. We then use the expert to generate a set of demonstrations, which will be used to train the DRL agent with BC.

To use BC, the first step is to generate some demonstrations with the expert:
```sh
python run_expert.py --train-dataset Jin2022 --train --valid --horizon 4 --proc-num 8
```
Check `run_expert.py` for the detailed explanations of the arguments.

Next, we can integrate BC into our tranining pipeline. Let's say pretrain the DRL agent with 150 BC steps:
```sh
python run_mansy.py --train --test --epoch 1000 --step-per-epoch 4096 --step-per-collect 4096 --lr 0.0005 --batch-size 512 --train --train-dataset Jin2022 --test --test-dataset Jin2022 --qoe-test-ids 0 1 2 3 --test-on-seen --lamb 0.5 --train-identifier --identifier-epoch 1000 --identifier-lr 0.0001 --device cuda:0 --gamma 0.95 --ent-coef 0.02 --seed 5 --use-identifier --bc --bc-max-steps 150 --bc-identifier-max-steps 150
```

Note: In our case, we do not find BC to work quite well (with negligible improvement on convergence speed or performance), so we do not report this trick in our paper.

## Citation
If you find this repository useful, please kindly cite our paper:
```
@article{duo2023mansy,
  title={MANSY: Generalizing Neural Adaptive Immersive Video Streaming With Ensemble and Representation Learning},
  author={Wu, Duo and Wu, Panlong and Zhang, Miao and Wang, Fangxin},
  journal={arXiv preprint arXiv:2311.06812},
  year={2023}
}
```