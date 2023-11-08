import argparse
import os
import math
import time
import random
import pickle
import torch
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from multiprocessing import Pool
from envs.parima_env import ParimaEnv
from utils.common import get_config_from_yml, generate_environment_samples, read_log_file, generate_environment_test_samples


def test(args, config, qoe_weights, results_dir):
    log_path = os.path.join(results_dir, f'results.csv')
    if os.path.exists(log_path):
        os.remove(log_path)

    videos = config.video_split[args.dataset]['test']
    users = config.user_split[args.dataset]['test']
    traces = config.network_split[args.network_dataset]['test']
    all_samples = generate_environment_test_samples(videos, users, traces, qoe_weights)
    
    parima_env = ParimaEnv(config, args.dataset, args.network_dataset, qoe_weights, all_samples,
                           log_path, config.startup_download, args.dataset_frequency, 'test', args.seed)
    sample_count = parima_env.sample_count()
    for i in tqdm(range(sample_count), desc='Testing: '):
        state = parima_env.reset()
        video, user, trace, qoe_weight = parima_env.current_video, parima_env.current_user, \
            parima_env.current_trace, parima_env.current_qoe_weight
        
        done = False
        while not done:
            bandwidth = parima_env.estimate_bandwidth(args.aggressive)
            action = parima_env.choose_action(bandwidth)
            state, reward, done, _ = parima_env.step(action)
        # print(f'video-{video}, user-{user}, trace-{trace}, qoe-weight-{qoe_weight} done!')
    
    read_log_file(log_path)
    print(log_path)


def run(args, config):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    split = 'train' if args.test_on_seen else 'test'
    if args.qoe_ids is None:
        args.qoe_ids = list(range(len(config.qoe_split[split])))

    if args.test_on_seen:
        results_dir = os.path.join(config.bs_results_dir, args.model,  args.dataset + '_' + args.network_dataset, 'seen_qoe' + '_'.join(map(str, args.qoe_ids)), f'seed_{args.seed}_agg_{args.aggressive}')
    else:
        results_dir = os.path.join(config.bs_results_dir, args.model,  args.dataset + '_' + args.network_dataset, 'unseen_qoe' + '_'.join(map(str, args.qoe_ids)), f'seed_{args.seed}_agg_{args.aggressive}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    qoe_weights = [config.qoe_split[split][i] for i in args.qoe_ids]
    print('Testing QoE weights:', qoe_weights)
    test(args, config, qoe_weights, results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='log_tensorboard')
    parser.add_argument('--model', type=str, default='parima')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='Wu2017')
    parser.add_argument('--dataset-frequency', type=int, default=5)
    parser.add_argument('--aggressive', type=float, default=1.0)
    parser.add_argument('--network-dataset', type=str, default='4G')
    parser.add_argument('--qoe-ids', type=int, nargs='*')  # on one or more qoe
    parser.add_argument("--test-on-seen", action='store_true')
    parser.add_argument('--proc-num', type=int, help='Process number for multiprocessing')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_known_args()[0]

    # debug
    # args.qoe_ids = [0, 1]
    # args.mode = 'test'
    # args.dataset = 'Jin2022'
    # args.aggressive = 1.5

    # On Wu2017
    # python run_parima.py --qoe-ids 0 --test-on-seen --mode test --aggressive 1.5 --seed 1
    # python run_parima.py --qoe-ids 1 --test-on-seen --mode test --aggressive 1.5 --seed 1
    # python run_parima.py --qoe-ids 2 --test-on-seen --mode test --aggressive 1.5 --seed 1
    # python run_parima.py --qoe-ids 3 --test-on-seen --mode test --aggressive 1.5 --seed 1

    # On Jin2022
    # python run_parima.py --dataset Jin2022 --qoe-ids 0 --test-on-seen --mode test --aggressive 1.5 --seed 1
    # python run_parima.py --dataset Jin2022 --qoe-ids 1 --test-on-seen --mode test --aggressive 1.5 --seed 1
    # python run_parima.py --dataset Jin2022 --qoe-ids 2 --test-on-seen --mode test --aggressive 1.5 --seed 1
    # python run_parima.py --dataset Jin2022 --qoe-ids 3 --test-on-seen --mode test --aggressive 1.5 --seed 1

    print(args)
    
    config = get_config_from_yml()
    run(args, config)
