"""
This file is used to clean the HMD traces and simplify the data.
The method for cleaning and simplifying data is from [1].
The codes of this file are mostly from [2].

[1] Anh Nguyen and Zhisheng Yan. 2019. A Saliency Dataset for 360-Degree Videos. In 10th ACM Multimedia Systems
    Conference (MMSys ’19), June 18–21, 2019, Amherst, MA, USA. ACM, New York, NY, USA, 6 pages.
    https://doi.org/10.1145/3304109.3325820
[2] Yiyun Lu, Yifei Zhu, and Zhi Wang. 2022. Personalized 360-Degree Video Streaming: A Meta-Learning Approach.
    In Proceedings of the 30th ACM International Conference on Multimedia (MM ’22), October 10–14, 2022, Lisboa,
    Portugal. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3503161.3548047
"""

import os
import csv
import numpy as np
import pickle

from utils import get_config_from_yml
import head_orientation_lib


def preprocess_hmd_trace(dataset, config):
    """
    Transform the format of the viewport traces of a video (from Quaternion to X, Y coordinate on a plane)
    
    :param dataset: dataset name
    :param config: configuration
    """
    print(f'Preprocess viewports for dataset {dataset}.')
    raw_dataset_dir = os.path.join(config.raw_datasets_dir[dataset], 'viewports')
    dataset_dir = config.viewport_datasets_dir[dataset]

    if dataset == 'Wu2017':
        origin_video_num, origin_user_num = 9, 48  
        for i in range(1, origin_video_num + 1):
            for j in range(1, origin_user_num + 1):
                raw_data_path = os.path.join(raw_dataset_dir, str(j), f'video_{(i - 1)}.csv')
                raw_data = np.loadtxt(raw_data_path, delimiter=',', usecols=(1, 2, 3, 4, 5), dtype=str)  # playback time, quaternion
                raw_data = raw_data[1:, :].astype(np.float32)  # skip headline and convert data into float
                playback_time, quaternion = raw_data[:, 0], raw_data[:, 1:]
                zyxw_quaternion = np.stack([quaternion[:, 2], quaternion[:, 1], quaternion[:, 0], quaternion[:, 3]], axis=1)
                
                geoxy_viewports = []
                for k, item in enumerate(zyxw_quaternion):
                    vector = head_orientation_lib.extract_direction_dataset2(item)
                    theta, phi = head_orientation_lib.vector_to_ang(np.array(vector))
                    y, x = head_orientation_lib.ang_to_geoxy(theta, phi, 1, 1)  # y: height-axis, x: weight-axis
                    geoxy_viewports.append([playback_time[k], x, y])
                
                data_dir = os.path.join(dataset_dir, f'video{i}')
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                data_path = os.path.join(data_dir, f'user{j}.csv')
                np.savetxt(data_path, np.array(geoxy_viewports), fmt='%.6f', delimiter=',')
                print(data_path)
    elif dataset == 'Jin2022':
        origin_video_num, origin_user_num = 27, 100
        label = 0
        for j in range(1, origin_user_num + 1):
            user_raw_data_dir = os.path.join(raw_dataset_dir, str(j))
            files = os.listdir(user_raw_data_dir)
            if len(files) != origin_video_num or j == 51:
                continue
            label += 1
            for file in files:
                i = int(file.split('_')[2])
                raw_data = np.loadtxt(os.path.join(user_raw_data_dir, file), delimiter=',', usecols=(0, 1, 2), dtype=str)       
                geoxy_viewports = raw_data[1:, :].astype(np.float32)  # skip headline and convert data into float
                _, video_width, video_height = config.video_info[dataset][i]
                geoxy_viewports[:, 1] /= video_width
                geoxy_viewports[:, 2] /= video_height

                data_dir = os.path.join(dataset_dir, f'video{i}')
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                data_path = os.path.join(data_dir, f'user{label}.csv')
                np.savetxt(data_path, np.array(geoxy_viewports), fmt='%.6f', delimiter=',')
                print(data_path)


def simplify_hmd_trace(dataset, config, frequency=10):
    """
    Simplify viewport traces according to the specified sampling frequency (e.g., 5 Hz).

    :param dataset: dataset name
    :param frequency: sampling frequence
    :param config: configuration
    """
    print(f'Simplify viewport for dataset {dataset} with sampling frequency {frequency} Hz.')
    dataset_dir = config.viewport_datasets_dir[dataset]
    video_num, user_num = config.video_num[dataset], config.user_num[dataset]

    for i in range(1, video_num + 1):
        for j in range(1, user_num + 1):
            origin_data_path = os.path.join(dataset_dir, f'video{i}', f'user{j}.csv')
            origin_data = np.loadtxt(origin_data_path, delimiter=',', dtype=np.float32)
            simplify_data = []
            timestamp, gap = 0, 1 / frequency
            rela_time = origin_data[0][0]
            for row in origin_data:
                playback_time = (row[0] - rela_time) if dataset == 'Jin2022' else row[0]
                if int(playback_time) > 0 and timestamp == 0:  # filter out dirty data
                    continue
                if playback_time >= timestamp:
                    simplify_data.append(row)
                    timestamp += gap
            simplify_data = np.array(simplify_data)
            simplify_data_dir = os.path.join(dataset_dir, f'video{i}', f'{frequency}Hz')
            if not os.path.exists(simplify_data_dir):
                os.makedirs(simplify_data_dir)
            simplify_data_path = os.path.join(simplify_data_dir, f'simple_{frequency}Hz_user{j}.csv')
            simplify_data_path_npy = os.path.join(simplify_data_dir, f'simple_{frequency}Hz_user{j}.npy')
            np.savetxt(simplify_data_path, simplify_data, fmt='%.6f', delimiter=',')
            np.save(simplify_data_path_npy, simplify_data)
            print('Simplified file saved at:', simplify_data_path + '/npy')


if __name__ == "__main__":
    config = get_config_from_yml()
    # preprocess_hmd_trace('Wu2017', config)
    simplify_hmd_trace('Jin2022', config)
