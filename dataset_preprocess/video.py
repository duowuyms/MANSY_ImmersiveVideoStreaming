import json
import os
import argparse
import subprocess
import time

from multiprocessing import Pool
from utils import get_config_from_yml


def segment_video_into_chunks(video_path, chunk_path, rate, start, duration):
    decoding_result = subprocess.run(["ffmpeg", "-y",
                                      "-ss", f"{start}",
                                      "-t", f"{duration}",
                                      "-accurate_seek",
                                      "-i", video_path,
                                      "-c:v", "libx264",
                                      "-b:v", f'{rate}M',
                                      "-avoid_negative_ts", "1",
                                      chunk_path],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
    if decoding_result.returncode != 0:
        print("DECODING FAILED")
        print(decoding_result.stdout)
        print(decoding_result.stderr)
        exit()


def crop_chunk_into_tiles(chunk_path, tile_path_fmt, rate, tile_res, tile_num_width, tile_num_height):
    for h in range(tile_num_height):
        for w in range(tile_num_width):
            tile_id = h * tile_num_height + w
            tile_path = tile_path_fmt % tile_id
            decoding_result = subprocess.run(["ffmpeg", "-y",
                                              "-i", chunk_path,
                                              "-vf",
                                              f"crop={tile_res[0]}:{tile_res[1]}:{w * tile_res[0]}:{h * tile_res[1]}",
                                              "-b:v", f'{rate}M',
                                              tile_path],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             universal_newlines=True)
            if decoding_result.returncode != 0:
                print("DECODING FAILED")
                print(decoding_result.stdout)
                print(decoding_result.stderr)
                exit()


def preprocess_video_one_rate(dataset, raw_video_dataset_dir, video, rate, config):
    """
    Each video in the dataset is encoded into multiple bitrate versions.
    This function profile the information of a video encoded with a specific bitrate.

    :param dataset: dataset name
    :param raw_video_dataset_dir: video dataset directory
    :param video: video id
    :param rate: bitrate 
    :param config: configuration
    """
    video_path = os.path.join(raw_video_dataset_dir, f'video{video}', f'{video}-{rate}M.mp4')
    video_tmp_dir = os.path.join(raw_video_dataset_dir, 'tmp', f'video{video}', str(rate))
    if not os.path.exists(video_tmp_dir):
        os.makedirs(video_tmp_dir)
    video_length, video_width, video_height = config.video_info[dataset][video]
    # video_length = 10  # for testing
    tile_res = video_width // config.tile_num_width, video_height // config.tile_num_height
    print('Video processing info:')
    print('Video ID and rate:', video, rate)
    print('Video:', video_path)
    print('Video length:', video_length)
    print('Temporary dir:', video_tmp_dir)

    chunk_info = {}
    for chunk_id in range(video_length // config.chunk_length):
        # clear temporary dir to avoid collision
        for fname in os.listdir(video_tmp_dir):
            if fname.endswith(".mp4"):
                os.remove(os.path.join(video_tmp_dir, fname))

        # step #1: segment a video clip into chunk
        chunk_path = os.path.join(video_tmp_dir, f'{chunk_id}-{chunk_id + config.chunk_length}.mp4')
        segment_video_into_chunks(video_path, chunk_path, rate, start=chunk_id, duration=config.chunk_length)
        # step #2: crop a chunk into tiles
        tile_path_fmt = os.path.join(video_tmp_dir, f'{chunk_id}-{chunk_id + config.chunk_length}-%d.mp4')
        crop_chunk_into_tiles(chunk_path, tile_path_fmt, rate, tile_res, config.tile_num_width, config.tile_num_height)
        # step #3: evaluate the quality and size of each tile
        size_arr, quality_arr = [], []
        for tile_id in range(config.tile_total_num):
            tile_path = tile_path_fmt % tile_id
            size = os.path.getsize(tile_path)
            size_arr.append(size)
            quality = rate  # we represent quality as bitrate
            quality_arr.append(quality)
        chunk_info[chunk_id] = {'size': size_arr, 'quality': quality_arr}
        print(f'({video}, {rate}) Chunk #{chunk_id} done...')
    return rate, chunk_info


def preprocess_video(dataset, video, config):
    """
    Preprocess one video in the dataset
    The video is profiled as a manifest json file, which records its information.

    :param dataset: dataset name
    :param video: video id
    :param config: configuration
    """
    raw_video_dataset_dir = os.path.join(config.raw_datasets_dir[dataset], 'videos')
    manifest_dataset_dir = config.video_datasets_dir[dataset]

    video_length, video_width, video_height = config.video_info[dataset][video]
    rates = list(sorted(config.video_rates))
    # video_length = 10  # for testing
    tile_res = video_width // config.tile_num_width, video_height // config.tile_num_height
    print('Video ID:', video)
    print('Rates:', rates)
    print('Length, resolution:', f'{video_length}, {video_width}x{video_height}')
    print('Tile resolution', f'{tile_res[0]}x{tile_res[1]}')

    video_data = {'Video_Time': video_length, 'Chunk_Count': video_length // config.chunk_length, 'Chunk_Time': config.chunk_length,
                  'Available_Bitrates': rates}  # store the video info in a dict, which will be dumped into a json file

    start_time = time.time()
    # use multiprocess to speed up processing
    pool = Pool(processes=5)
    results = []
    for i in range(len(rates)):
        results.append(pool.apply_async(preprocess_video_one_rate, (dataset, raw_video_dataset_dir, video, rates[i], config,)))
    pool.close()
    pool.join()

    chunk_info_each_rate = {}
    for result in results:
        rate, chunk_info = result.get()
        chunk_info_each_rate[rate] = chunk_info

    # reorganize chunk information at each rate
    reorg_chunk_info = {}
    for chunk_id in range(video_length // config.chunk_length):
        size_each_rate, quality_each_rate = [], []
        for rate in rates:
            size_each_rate.append(chunk_info_each_rate[rate][chunk_id]['size'])
            quality_each_rate.append(chunk_info_each_rate[rate][chunk_id]['quality'])
        reorg_chunk_info[chunk_id] = {'size': size_each_rate, 'quality': quality_each_rate}

    end_time = time.time()
    video_data['Chunks'] = reorg_chunk_info
    manifest = os.path.join(manifest_dataset_dir, f'video{video}.json')
    json.dump(video_data, open(manifest, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'Manifest file of video {video} is saved at {manifest}.')
    print(f'Runtime: {round((end_time - start_time) / 3600, 2)}h')


def preprocess_video_dataset(dataset, config):
    """
    Preprocess video dataset. 

    :param dataset: dataset name
    :param config: configuration
    """
    print(f'Preprocess videos for dataset {dataset}.')

    video_num = config.video_num[dataset]
    for video in range(1, video_num + 1):
        preprocess_video(dataset, video, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Jin2022')
    args = parser.parse_known_args()[0]

    config = get_config_from_yml()
    preprocess_video_dataset(args.dataset, config)