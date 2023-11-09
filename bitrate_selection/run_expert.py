import argparse
import os
import copy
import math
import time
import random
import pickle
import torch
import numpy as np
from prettytable import PrettyTable
from multiprocessing import Pool
from tianshou.data import ReplayBuffer, Batch
from envs.expert_env import ExpertEnv
from utils.common import get_config_from_yml, generate_environment_samples, read_log_file, generate_environment_test_samples


def collect_demonstrations(expert_env):
    sample_count = expert_env.sample_count()
    demonstrations = {}
    for _ in range(sample_count):
        state = expert_env.reset()
        video, user, trace, qoe_weight = expert_env.current_video, expert_env.current_user, \
            expert_env.current_trace, expert_env.current_qoe_weight
        qoe_weight = tuple(qoe_weight.astype(np.int32).tolist())
        
        states, actions = [], []
        done = False
        while not done:
            action = expert_env.choose_action()
            states.append(copy.deepcopy(state))
            actions.append(action)
            state, reward, done, _ = expert_env.step(action)
        
        length = len(states)
        demonstration = ReplayBuffer(length)
        for i in range(length):
            demonstration.add(Batch(obs=states[i], act=actions[i], rew=0, done=(i == length - 1),
                                    obs_next=i + 1, info={}))
        demonstrations[video, user, trace, qoe_weight] = demonstration
        print(f'Demonstration of video-{video}, user-{user}, trace-{trace}, qoe-weight-{qoe_weight} done!')
    return demonstrations


def create_demonstrations(args, config, qoe_weights, models_dir, demos_dir, cache_path, mode='train'):
    log_path = os.path.join(models_dir, f'{mode}_log.csv')
    demo_path = os.path.join(demos_dir, f'{mode}_demonstrations.pkl')
    if os.path.exists(log_path):
        os.remove(log_path)
    
    videos = config.video_split[args.train_dataset][mode]
    users = config.user_split[args.train_dataset][mode]
    traces = config.network_split[args.network_dataset][mode]
    all_samples = generate_environment_samples(videos, users, traces, qoe_weights, seed=args.seed)
    print('Total samples:', len(all_samples))
    
    proc_num = args.proc_num
    samples_per_proc = math.ceil(len(all_samples) / proc_num)
    expert_envs = []
    for i in range(proc_num):
        samples = []
        for j in range(i * samples_per_proc, min((i + 1) * samples_per_proc, len(all_samples))):
            samples.append(all_samples[j])
        expert_envs.append(ExpertEnv(config, args.train_dataset, args.network_dataset, qoe_weights, samples,
                                     demos_dir, cache_path, log_path, config.startup_download, args.horizon,
                                     args.refresh_cache if i == 0 else False, mode, args.seed))
    
    start_time = time.time()
    pool = Pool(proc_num)
    results = []
    for i in range(proc_num):
        results.append(pool.apply_async(collect_demonstrations, (expert_envs[i],)))
    pool.close()
    pool.join()

    total_demonstrations = {}
    for result in results:
        total_demonstrations.update(result.get())
    end_time = time.time()

    pickle.dump(total_demonstrations, open(demo_path, 'wb'))
    print(f'Create {len(all_samples)} demonstrations, saved at {demo_path}, cost {round((end_time - start_time) / 3600, 2)}h')
    

def test(args, config, qoe_weights, results_dir, demos_dir, cache_path):
    log_path = os.path.join(results_dir, 'results.csv')
    if os.path.exists(log_path):
        os.remove(log_path)

    videos = config.video_split[args.test_dataset]['test']
    users = config.user_split[args.test_dataset]['test']
    traces = config.network_split[args.network_dataset]['test']
    all_samples = generate_environment_test_samples(videos, users, traces, qoe_weights)
    
    expert_env = ExpertEnv(config, args.test_dataset, args.network_dataset, qoe_weights, all_samples,
                           demos_dir, cache_path, log_path, config.startup_download, args.horizon,
                           args.refresh_cache, 'test', args.seed)
    sample_count = expert_env.sample_count()
    for i in range(sample_count):
        state = expert_env.reset()
        video, user, trace, qoe_weight = expert_env.current_video, expert_env.current_user, \
            expert_env.current_trace, expert_env.current_qoe_weight
        
        done = False
        while not done:
            action = expert_env.choose_action()
            state, reward, done, _ = expert_env.step(action)
        print(f'video-{video}, user-{user}, trace-{trace}, qoe-weight-{qoe_weight} done!')
    
    read_log_file(log_path)


def run(args, config):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    if args.qoe_train_ids is None:
        args.qoe_train_ids = list(range(len(config.qoe_split['train'])))
    split = 'train' if args.test_on_seen else 'test'
    if args.qoe_test_ids is None:
        args.qoe_test_ids = list(range(len(config.qoe_split[split])))

    models_dir = os.path.join(config.bs_models_dir, args.model, args.train_dataset + '_' + args.network_dataset, 'qoe' + '_'.join(map(str, args.qoe_train_ids)))
    demos_dir = os.path.join(config.bs_models_dir, args.model, args.train_dataset + '_' + args.network_dataset, 'qoe' + '_'.join(map(str, args.qoe_train_ids)))
    if args.test_on_seen:
        results_dir = os.path.join(config.bs_results_dir, args.model,  args.test_dataset + '_' + args.network_dataset, 'seen_qoe' + '_'.join(map(str, args.qoe_test_ids)))
    else:
        results_dir = os.path.join(config.bs_results_dir, args.model,  args.test_dataset + '_' + args.network_dataset, 'unseen_qoe' + '_'.join(map(str, args.qoe_test_ids)))
    train_cache_path = os.path.join(config.bs_models_dir, args.model, f'{args.train_dataset}_cache.pkl')
    test_cache_path = os.path.join(config.bs_models_dir, args.model, f'{args.test_dataset}_cache.pkl')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(demos_dir):
        os.makedirs(demos_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if args.train:
        qoe_weights = [config.qoe_split['train'][i] for i in args.qoe_train_ids]
        print("Training QoE weights:", qoe_weights)
        create_demonstrations(args, config, qoe_weights, models_dir, demos_dir, train_cache_path, 'train')
    if args.valid:
        qoe_weights = [config.qoe_split['valid'][i] for i in args.qoe_train_ids]
        print('Validating QoE weights:', qoe_weights)
        create_demonstrations(args, config, qoe_weights, models_dir, demos_dir, train_cache_path, 'valid')
    if args.test:
        qoe_weights = [config.qoe_split[split][i] for i in args.qoe_test_ids]
        print('Testing QoE weights:', qoe_weights)
        test(args, config, qoe_weights, results_dir, demos_dir, test_cache_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='log_tensorboard')
    parser.add_argument('--model', type=str, default='expert')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--valid", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--test-on-seen", action='store_true')
    parser.add_argument('--train-dataset', type=str, default='Wu2017')
    parser.add_argument('--test-dataset', type=str, default='Wu2017')
    parser.add_argument('--network-dataset', type=str, default='4G')
    parser.add_argument('--qoe-train-ids', type=int, nargs='*')  # train on one or more qoe
    parser.add_argument('--qoe-test-ids', type=int, nargs='*')  # test on one or more qoe
    parser.add_argument('--proc-num', type=int, help='Process number for multiprocessing')
    parser.add_argument('--horizon', type=int, help='The horizon for expert to look ahead', default=4)
    parser.add_argument('--refresh-cache', action='store_true', help='Refresh cache')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_known_args()[0]

    # Train/Valid
    # python run_expert.py --train-dataset Jin2022 --train --valid --horizon 4 --proc-num 8

    # Test
    # python run_expert.py --test-dataset Jin2022 --test --horizon 2 --qoe-test-ids 3 --test-on-seen

    print(args)
    
    config = get_config_from_yml()
    run(args, config)
