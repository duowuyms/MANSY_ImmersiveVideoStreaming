import os
import argparse
import random
import sys
import pickle
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from models import LinearRegression, ViewportTransformerMTIO
from utils.common import get_config_from_yml, find_tiles_covered_by_viewport, find_block_covered_by_point
from utils.load_dataset import create_dataset


def predict(args, config, model, videos, users, dataloader, results_dir, model_path):
    if args.model != 'regression':  # linear regression doesn't need loading model weights
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        print('Successfully loaded model from', model_path)

    print(f'Predict with model {args.model} on {args.dataset} - seed: {args.seed}')
    with torch.no_grad():
        model.eval()
        results = {(video, user): [] for video in videos for user in users}
        for batch, (history, current, future, video, user, timesteps) in enumerate(tqdm(dataloader)):
            history, current, future = history.to(args.device), current.to(args.device), future.to(args.device)
            batch_size = history.shape[0]
            pred, gt = model.sample(history, current), future
            merge = torch.cat([gt, pred], dim=-1).cpu()
            for i in range(batch_size):
                results[int(video[i]), int(user[i])].append(np.array(merge[i]))

    print(f'Process prediction data')
    data = {}
    for key, value in results.items():
        viewports = []
        for i in range(len(value)):
            gt_viewport = np.zeros(config.tile_total_num, dtype=np.uint8)
            pred_viewport = np.zeros(config.tile_total_num, dtype=np.uint8)
            for j in range(args.dataset_frequency):
                gt_x, gt_y = int(value[i][j][0] * config.video_width), int(value[i][j][1] * config.video_height)
                gt_viewport |= find_tiles_covered_by_viewport(gt_x, gt_y, config.video_width, config.video_height, config.tile_width, 
                                                              config.tile_height, config.tile_num_width, config.tile_num_height).reshape(-1)
                pred_x, pred_y = int(value[i][j][2] * config.video_width), int(value[i][j][3] * config.video_height)
                pred_viewport |= find_tiles_covered_by_viewport(pred_x, pred_y, config.video_width, config.video_height, config.tile_width,
                                                                config.tile_height, config.tile_num_width, config.tile_num_height).reshape(-1)
            accuracy = np.sum(gt_viewport & pred_viewport) / np.sum((gt_viewport | pred_viewport))  # IoU
            viewports.append((i + args.trim_head // args.dataset_frequency, gt_viewport, pred_viewport, accuracy))
        data[key] = viewports
    
    for key, value in data.items():
        video, user = key
        base_dir = os.path.join(results_dir, f'video{video}')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        pkl_path = os.path.join(base_dir, f'user{user}.pkl')
        csv_path = os.path.join(base_dir, f'user{user}.csv')
        pickle.dump(value, open(pkl_path, 'wb'))
        with open(csv_path, 'w', encoding='utf-8') as file:
            head = 'chunk,gt,pred,accuracy\n'
            file.write(head)
            for i in range(len(value)):
                gt_viewport = ','.join(map(str, list(value[i][1])))
                pred_viewport = ','.join(map(str, list(value[i][2])))
                file.write(f'{value[i][0]},{gt_viewport},{pred_viewport},{value[i][3]}\n')
            file.close()


def create_model(model_name, fut_window, hidden_dim, block_num, device, seed):
    model = None
    if model_name == 'regression':
        model = LinearRegression(fut_window=fut_window, device=device)
    elif model_name == 'mtio':
        model = ViewportTransformerMTIO(in_channel=2, fut_window=fut_window, d_model=hidden_dim, dim_feedforward=hidden_dim,
                                        num_encoder_layers=block_num, num_decoder_layers=block_num, device=device, seed=seed)
    return model


def run(args, config):
    assert args.model in ['regression', 'mtio']

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    results_dir = os.path.join(config.viewport_datasets_dir[args.dataset], 'prediction')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model = create_model(args.model, args.fut_window, args.hidden_dim, args.block_num, args.device, args.seed).to(args.device)
    
    if args.compile:
        assert torch.__version__ >= '2.0.0', 'Compile model requires torch version >= 2.0.0, but current torch version is ' + torch.__version__
        print("\033[33mWarning:\033[0m There seems to be some bugs in torch.compile. If batch size is too large, it will raise errors (I don't know why this happens).")
        model = torch.compile(model).to(args.device)  # recommend to compile model when you are using PyTorch 2.0
    
    torch.set_float32_matmul_precision('high')

    dataset_videos, dataset_users = [], []
    for split in ['train', 'valid', 'test']:  
        dataset_videos += config.video_split[args.dataset][split]
        dataset_users += config.user_split[args.dataset][split]
    dataset_videos, dataset_users = list(set(dataset_videos)), list(set(dataset_users))
    dataset = create_dataset(args.dataset, config, his_window=args.his_window, fut_window=args.fut_window, sample_step=args.sample_step,
                             frequency=args.dataset_frequency, trim_head=args.trim_head, trim_tail=args.trim_tail,
                             dataset_video_split={'merge': dataset_videos}, dataset_user_split={'merge': dataset_users},
                             include=['merge'])[0]
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False, pin_memory=True)
    predict(args, config, model, dataset_videos, dataset_users, dataloader, results_dir, args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the input parameters to train the network.')

    # ========== model/plm settings related arguments ==========
    parser.add_argument('--device', action='store', dest='device', help='Device (cuda or cpu) to run experiment.')
    parser.add_argument('--model', action='store', dest='model', help='Model type')
    parser.add_argument('--hidden-dim', type=int, help='Number of hidden dimensions')
    parser.add_argument('--block-num', type=int, help='Number of encoder/decoder/LSTM/GRU blocks')
    parser.add_argument('--model-path', action='store', dest='model_path', help='Path to load model')
    parser.add_argument('--compile', action='store_true', dest='compile', 
                        help='(Optional) Compile model for speed up (available only for PyTorch 2.0).')
    
    # ========== dataset loading/processing settings related arguments ==========
    parser.add_argument('--dataset', action='store', dest='dataset', help='Dataset for prediction.')
    parser.add_argument('--his-window', action='store', dest='his_window', help='Historical window', type=int)
    parser.add_argument('--fut-window', action='store', dest='fut_window', help='Future (prediction) window (default 10).', type=int)
    parser.add_argument('--trim-head', action='store', dest='trim_head',
                        help='(Optional) Trim some part of the viewport trajectory head (default 30).', type=int)
    parser.add_argument('--trim-tail', action='store', dest='trim_tail',
                        help='(Optional) Trim some part of the viewport trajectory tail (default 10).', type=int)
    parser.add_argument('--dataset-frequency', action='store', dest='dataset_frequency',
                        help='(Optional) The frequency version of the dataset (default 5).', type=int)
    parser.add_argument('--sample-step', action='store', dest='sample_step',
                        help='(Optional) The steps for sampling viewports (default 5).', type=int)
    parser.add_argument('--bs', action="store", dest='bs', help='Neural network batch size.', type=int)
    parser.add_argument('--seed', action="store", dest='seed', type=int, default=5,
                        help='(Optional) Random seed (default to 1).')
    
    
    args = parser.parse_args()

    # command example
    # python predict.py --model regression --device cpu --dataset Jin2022 --bs 64 --seed 1

    # handle defautl settings
    config = get_config_from_yml()
    args.trim_head = config.trim_head if args.trim_head is None else args.trim_head
    args.trim_tail = config.trim_tail if args.trim_tail is None else args.trim_tail
    args.dataset_frequency = config.frequency if args.dataset_frequency is None else args.dataset_frequency
    args.sample_step = config.sample_step if args.sample_step is None else args.sample_step

    if args.model == 'regression':
        args.train = False
        args.compile = False
        args.device = 'cpu'
        print('Detect model: regression. Automatically disenable train and compile mode and set device to cpu.')
        
    print(args)
    run(args, config)
