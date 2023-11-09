import os
import json
import numpy as np
from simulators.buffer import PlaybackBuffer
from simulators.hmdtrace import HMDTrace
from simulators.network import NetworkTrace


class Simulator:
    """
    Given specific video, hmd trace and network trace, this class simulate the whole video watching
    and downloading process
    """

    def __init__(self, config, dataset, video, user, network_dataset, trace, startup_download, trace_scale=None, device='cpu'):
        """
        param config: configuration instance
        dataset: viewport/video dataset
        video: video id
        user: user id
        network_dataset: network dataset
        trace: network trace id
        startup_download: downloaded chunks for starting a streaming session
        trace_scale: whether to scale network trace, if so, provide (up, low) for scaling
        """
        self.config = config
        self.startup_download = startup_download
        self.buffer = PlaybackBuffer(startup_download, config.chunk_length)

        viewport_path = os.path.join(config.viewport_datasets_dir[dataset], 'prediction',f'video{video}',  f'user{user}.pkl')
        self.hmd_trace = HMDTrace(viewport_path, config.tile_num_width, config.tile_num_height)

        trace_path = os.path.join(config.network_datasets_dir[network_dataset], config.network_info[network_dataset][trace])
        self.net_trace = NetworkTrace(trace_path, scale=trace_scale)

        manifest_path = os.path.join(config.video_datasets_dir[dataset], f'video{video}.json')
        self.video_manifest = json.load(open(manifest_path, 'r', encoding='utf-8'))
        self.video_length = self.video_manifest['Video_Time']
        self.chunk_info = self.video_manifest['Chunks']

        self.start_chunk, self.end_chunk, self.chunk_num = self.hmd_trace.get_hmd_trace_info()
        self.end_chunk = min(self.end_chunk, self.video_length - 1)
        self.chunk_num = self.end_chunk - self.start_chunk + 1
        assert startup_download + 1 >= self.start_chunk
        self.next_chunk = startup_download + 1
        self.device = device

    def get_next_chunk_size(self, chunk=None):
        """
        Get the sizes of tiles of next chunk at different bitrates
        """
        if chunk is None:  # by default, chunk id is the next chunk
            chunk = self.next_chunk
        return np.array(self.chunk_info[str(chunk)]['size'], dtype=np.float32)
    
    def get_chunk_num(self):
        """
        Get the chunk number of the video
        """
        return self.chunk_num

    def get_next_chunk_quality(self, chunk=None):
        """
        Get the quality of tiles of next chunk at different bitrates
        """
        if chunk is None:  # by default, chunk id is the next chunk
            chunk = self.next_chunk
        return np.array(self.chunk_info[str(chunk)]['quality'], dtype=np.float32)

    def get_next_chunk_info(self, chunk=None):
        if chunk is None:  # by default, chunk id is the next chunk
            chunk = self.next_chunk
        chunk_info = self.chunk_info[str(chunk)]
        return chunk_info['size'], chunk_info['quality']

    def get_viewport(self, chunk=None, flatten=True):
        if chunk is None:  # by default, chunk id is the next chunk
            chunk = self.next_chunk
        gt_viewport, pred_viewport, accuracy = self.hmd_trace.get_viewport(chunk, flatten=flatten)
        return np.array(gt_viewport, dtype=np.float32), np.array(pred_viewport, dtype=np.float32), accuracy

    def get_buffer_size(self):
        return self.buffer.get_buffer_size()

    def get_next_chunk(self):
        return self.next_chunk

    def simulate_download(self, tile_rates):
        """
        Given the bitrates of each tile, simulate the download of next chunk
        :param tile_rates: bitrates of each tile
        """
        selected_tile_sizes, selected_tile_quality = [], []
        size_info = self.chunk_info[str(self.next_chunk)]['size']
        quality_info = self.chunk_info[str(self.next_chunk)]['quality']
        for tile_id in range(self.config.tile_total_num):
            rate = tile_rates[tile_id]
            selected_tile_sizes.append(size_info[rate][tile_id])
            selected_tile_quality.append(quality_info[rate][tile_id])
        chunk_size = sum(selected_tile_sizes)
        chunk_quality = sum(selected_tile_quality)
        download_time = self.net_trace.simulate_download(chunk_size)
        rebuffer_time = self.buffer.push_chunk(self.config.chunk_length, download_time)
        actual_viewport, *_ = self.hmd_trace.get_viewport(self.next_chunk, flatten=True)
        self.next_chunk += 1
        over = self.next_chunk > self.end_chunk
        return np.array(selected_tile_sizes, dtype=np.float32), np.array(selected_tile_quality, dtype=np.float32),\
               chunk_size, chunk_quality, download_time, rebuffer_time, actual_viewport, over
    
    def reset(self):
        self.buffer.reset()
        self.hmd_trace.reset()
        self.net_trace.reset()
        self.next_chunk = self.startup_download


class ExpertSimulator(Simulator):
    def __init__(self, config, dataset, video, user, network_dataset, trace, startup_download, trace_scale=None, device='cpu'):
        super().__init__(config, dataset, video, user, network_dataset, trace, startup_download, trace_scale, device)
        
        self.record_net_trace_cur_idx = 0
        self.record_net_trace_cur_time = 0
        self.record_buf_size = 0
    
    def virtual_simulate_download_with_chunk_size(self, chunk_size, start: bool, end: bool):
        """
        Virtually download a chunk, just used by the expert.
        At the start of each virtual download, we need to record the buffer size, network trace information.
        At the end of each virtual download, we need to recover the buffer size, network trace information.
        """
        if start:
            self.record_buf_size = self.buffer.get_buffer_size()
            self.record_net_trace_cur_idx, self.record_net_trace_cur_time = self.net_trace.get_current_time()

        download_time = self.net_trace.simulate_download(chunk_size)
        rebuffer_time = self.buffer.push_chunk(self.config.chunk_length, download_time)
        self.next_chunk += 1
        over = self.next_chunk > self.end_chunk

        if end:
            self.buffer.set_buffer_size(self.record_buf_size)
            self.net_trace.set_current_time(self.record_net_trace_cur_idx, self.record_net_trace_cur_time)

        return download_time, rebuffer_time, over
    
    def calculate_chunk_size_and_quality(self, chunk, tile_rates, actual_viewport):
        selected_tile_sizes, selected_tile_quality = [], []
        size_info = self.chunk_info[str(chunk)]['size']
        quality_info = self.chunk_info[str(chunk)]['quality']
        for tile_id in range(self.config.tile_total_num):
            rate = tile_rates[tile_id]
            selected_tile_sizes.append(size_info[rate][tile_id])
            selected_tile_quality.append(quality_info[rate][tile_id])
        selected_tile_quality = np.array(selected_tile_quality, dtype=np.float32)
        chunk_size = sum(selected_tile_sizes)
        viewport_quality = sum(actual_viewport * selected_tile_quality) / sum(actual_viewport)
        return chunk_size, viewport_quality, selected_tile_quality
        
