import os
import numpy as np
from torch.utils.data import Dataset


class ViewportDataset(Dataset):
    """
    Wrapper class for viewport dataset.
    """
    def __init__(self, total_traces, videos, users, his_window, fut_window, trim_head, trim_tail, step):
        """
        :param total_traces: total viewport traces
        :param videos: video list
        :param users: user list
        :param his_window: historical window
        :param fut_window: future (prediction) window
        :param trim_head: trim some part of the viewport trajectory head
        :param trim_tail: trim some part of the viewport trajectory tail
        :param step: step size of sliding prediction window
        """
        self.total_traces = total_traces
        self.videos = videos
        self.users = users
        self.history_window = his_window
        self.future_window = fut_window
        self.trim_head = trim_head
        self.trim_tail = trim_tail
        self.step = step

        # total_traces store the viewport trace of each video and each user
        # we create a list trace_indices to record the indices to the samples in the traces of specific videos and users
        # the idea here is inspired by Quentin Guimard's repo: https://gitlab.com/DVMS_/DVMS
        self.trace_indices = []
        for video in videos:
            for user in users:
                trace = self.total_traces[video][user]
                for timestep in range(self.trim_head, len(trace) - self.trim_tail, self.step):
                    self.trace_indices.append((video, user, timestep))

    def __len__(self):
        return len(self.trace_indices)

    def __getitem__(self, index):
        """
        With index and self.trace_indices, we can easily access a specific viewport trajectory in the dataset.
        This method is implemented by subclass ViewportDataset360 and ViewportDatasetVV.
        """
        video, user, timestep = self.trace_indices[index]
        history = self.total_traces[video][user][timestep - self.history_window:timestep]
        current = self.total_traces[video][user][timestep:timestep + 1]
        future = self.total_traces[video][user][timestep + 1:timestep + self.future_window + 1]
        return history, current, future, video, user, timestep


def pack_data(dataset_dir, video_user_pairs, frequency):
    """
    Pack the viewport traces and video content features of corresponding video and user pairs
    into easy-access dict objects
    :param dataset_dir: directory of dataset
    :param video_user_pairs: list of video-user pairs
    :param frequency: the frequency version of the dataset
    :return: total_traces, total_content_features
    """
    pack_traces = {video: {} for video, _ in video_user_pairs}
    for video, user in video_user_pairs:
        data_path = os.path.join(dataset_dir, f'video{video}', f'{frequency}Hz', f'simple_{frequency}Hz_user{user}.npy')
        data = np.load(data_path)
        pack_traces[video][user] = data[:, 1:]  # the first column (i.e., column = 0) is timestep, we don't need it
    return pack_traces


def create_dataset(dataset, config, his_window, fut_window, trim_head=None, trim_tail=None, frequency=None, sample_step=None, 
                   dataset_video_split=None, dataset_user_split=None, include=['train', 'valid', 'test', 'test_seen', 'test_unseen']):
    """
    Create dataset.
    :param dataset: dataset name
    :param config: configuration
    :param his_window: historical window
    :param fut_window: future (prediction) window
    :param trim_head: trim some part of the viewport trajectory head
    :param trim_tail: trim some part of the viewport trajectory tail
    :param frequency: the frequency version of the dataset
    :param sample_step: the step for sampling viewports
    :param dataset_video_split: train, valid, test split info of videos
    :param dataset_user_split: train, valid, test split info of users
    :param include: inclusion of the splits of dataset
    :return: dataset_train, dataset_valid, dataset_test
    """
    dataset_dir = config.viewport_datasets_dir[dataset]
    # handling default settings
    if trim_head is None:
        trim_head = config.trim_head
    if trim_tail is None:
        trim_tail = config.trim_tail
    if frequency is None:
        frequency = config.frequency
    if sample_step is None:
        sample_step = config.sample_step
    if dataset_video_split is None:
        dataset_video_split = dict(config.video_split[dataset])
    if dataset_user_split is None:
        dataset_user_split = dict(config.user_split[dataset])
    
    if 'test_seen' in include:
        dataset_video_split['test_seen'] = dataset_video_split['test']
        min_length = min(len(dataset_user_split['valid']), len(dataset_user_split['test']))
        dataset_user_split['test_seen'] = dataset_user_split['valid'][:min_length]
    if 'test_unseen' in include:
        dataset_video_split['test_unseen'] = dataset_video_split['test']
        min_length = min(len(dataset_user_split['valid']), len(dataset_user_split['test']))
        dataset_user_split['test_unseen'] = dataset_user_split['test'][:min_length]

    total_video_user_pairs = set()
    for split in include:
        videos = dataset_video_split[split]
        users = dataset_user_split[split]
        for video in videos:
            for user in users:
                total_video_user_pairs.add((video, user))
    total_video_user_pairs = list(total_video_user_pairs)
    total_traces = pack_data(dataset_dir, total_video_user_pairs, frequency)
    dataset_splits = []
    for split in include:
        dataset_splits.append(
            ViewportDataset(total_traces, dataset_video_split[split],
                            dataset_user_split[split], his_window, fut_window, trim_head, trim_tail, sample_step)
        )
    return dataset_splits

