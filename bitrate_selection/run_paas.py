import argparse
import os
import pprint
import sys
import time
import gym
import random
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Batch, Collector, VectorReplayBuffer, ReplayBuffer, AsyncCollector, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv, RayVectorEnv
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from envs.paas_env import PAASEnv
from models.pass_dqn import DQNPolicy
from models.paas import PAASNet
from utils.common import get_config_from_yml, read_log_file


def train(args, config, policy, qoe_weights, models_dir, file_prefix):
    train_log_path = os.path.join(models_dir, file_prefix + '_train_log.csv')
    valid_log_path = os.path.join(models_dir, file_prefix + '_valid_log.csv')
    if os.path.exists(train_log_path):
        os.remove(train_log_path)
    if os.path.exists(valid_log_path):
        os.remove(valid_log_path)

    env = PAASEnv(config, args.train_dataset, args.network_dataset, qoe_weights, valid_log_path, config.startup_download,
                  mode='valid', seed=args.seed, device=args.device)
    env.seed(args.seed)

    args.train_num = len(qoe_weights) * 2
    args.test_num = len(qoe_weights) 
    args.episode_per_test = env.sample_count()
    print('Training num:', args.train_num)
    print('Test num:', args.test_num)
    print('Episode per test:', args.episode_per_test)

    train_env = SubprocVectorEnv(
        [lambda: PAASEnv(config, args.train_dataset, args.network_dataset, qoe_weights, train_log_path, config.startup_download,
                         mode='train', seed=args.seed, worker_num=args.train_num, device=args.device) for _ in range(args.train_num)],

    )
    valid_env = DummyVectorEnv(
        [lambda: PAASEnv(config, args.train_dataset, args.network_dataset, qoe_weights, valid_log_path, config.startup_download,
                         mode='valid', seed=args.seed, worker_num=args.test_num, device=args.device) for _ in range(args.test_num)],
    )
    print('Training QoE weights:', qoe_weights)
    
    train_env.seed(args.seed)
    valid_env.seed(args.seed)

    def dist(logits):
        return Categorical(probs=F.softmax(logits, dim=-1))
    
    checkpoint_path = os.path.join(models_dir, file_prefix + '_checkpoint.pth')
    best_policy_path = os.path.join(models_dir, file_prefix + '_best_policy.pth')
    if args.resume:
        if os.path.exists(checkpoint_path):
            policy.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
            print("Successfully loaded agent from:", checkpoint_path)
        else:
            print('Failed to load agent:', checkpoint_path, 'no such file')
    
    def save_best_fn(policy):
        torch.save(policy.state_dict(), best_policy_path)
        print('====================================================================')
        print('Best policy save at ' + best_policy_path)
        print('====================================================================')

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(policy.state_dict(), checkpoint_path)
        print('====================================================================')
        print('Checkpoint saved at ' + checkpoint_path)
        print('====================================================================')
        return checkpoint_path

    def train_fn(epoch, env_step):
        # eps annealing, just a demo
        if env_step <= 250000:
            policy.set_eps(args.eps_train)
        elif env_step <= 500000:
            eps = args.eps_train - (env_step - 250000) / \
                  500000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    # policy.set_eps(0.1)

    # collector
    # replay_buffer = ReplayBuffer(args.buffer_size)
    # replay_buffer = VectorReplayBuffer(
    #     total_size=args.buffer_size // len(train_env),
    #     buffer_num=len(train_env),
    # )
    replay_buffer = PrioritizedVectorReplayBuffer(
        alpha=0.5,
        beta=0.5,
        total_size=args.buffer_size // len(train_env),
        buffer_num=len(train_env),
    )
    train_collector = Collector(
        policy=policy, env=train_env, buffer=replay_buffer
    )
    log_path = os.path.join(models_dir, file_prefix + '_pass_tb_logger')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    valid_collector = Collector(policy, valid_env)

    trainer = OffpolicyTrainer(
        policy,
        train_collector,
        valid_collector,
        args.epochs,
        args.step_per_epoch,
        args.step_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        update_per_epoch=args.update_per_step,
        train_fn=train_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        resume_from_log=args.resume,
        logger=logger,
    )
    
    for epoch, epoch_stat, info in trainer:
        print(f"Epoch: {epoch}")
        print(epoch_stat)
        print(info)


def test(args, config, policy, qoe_weights, models_dir, results_dir, file_prefix):
    test_log_path = os.path.join(results_dir, file_prefix + '_results.csv')
    if os.path.exists(test_log_path):
        os.remove(test_log_path)

    test_env = PAASEnv(config, args.test_dataset, args.network_dataset, qoe_weights, test_log_path, config.startup_download,
                       mode='test', seed=args.seed, device=args.device)
    test_env.seed(args.seed)

    policy_path = args.policy_path
    if policy_path is None:
        policy_path = os.path.join(models_dir, file_prefix + '_best_policy.pth')
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
        print("Successfully loaded agent from:", policy_path)
    else:
        raise FileExistsError(f"File not exist: {policy_path}")

    with torch.no_grad():
        state = test_env.reset()
        sample_count = test_env.sample_count()
        for i in tqdm(range(sample_count), desc='Testing: '):
            done = False
            while not done:
                for key, value in state.items():
                    state[key] = np.expand_dims(value, 0)
                batch = {'obs': state}
                batch = Batch(obs=Batch(batch), info={})
                # logits = policy(batch, state=None).logits
                # action = F.softmax(logits, dim=-1).argmax().item()
                action = policy(batch, state=None).act
                action = action.item()
                state, reward, done, _ = test_env.step(action)
            state = test_env.reset()
        read_log_file(test_log_path)
        print('Results saved at:', test_log_path)

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
    if args.test_on_seen:
        results_dir = os.path.join(config.bs_results_dir, args.model,  args.test_dataset + '_' + args.network_dataset, 'seen_qoe' + '_'.join(map(str, args.qoe_test_ids)))
    else:
        results_dir = os.path.join(config.bs_results_dir, args.model,  args.test_dataset + '_' + args.network_dataset, 'unseen_qoe' + '_'.join(map(str, args.qoe_test_ids)))
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    file_prefix = f'epochs_{args.epochs}_bs_{args.batch_size}_lr_{args.lr}_gamma_{args.gamma}_seed_{args.seed}_tuf_{args.target_update_freq}'

    model = PAASNet(config, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    policy = DQNPolicy(
        model,
        optimizer,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)

    torch.set_float32_matmul_precision('high')

    if args.train:
        qoe_weights = [config.qoe_split['train'][i] for i in args.qoe_train_ids]
        train(args, config, policy, qoe_weights, models_dir, file_prefix)
    if args.test:
        qoe_weights = [config.qoe_split[split][i] for i in args.qoe_test_ids]
        test(args, config, policy, qoe_weights, models_dir, results_dir, file_prefix)


if __name__ == '__main__':
    # python run_PAAS.py --seed 1 --lr 0.0001 --gamma 0.99 --batch-size 256 --device cuda:0 --model paas1 --mode train
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PAAS')
    parser.add_argument('--reward-threshold', type=float, default=500000.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eps-test', type=float, default=0.0)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=50)
    parser.add_argument('--step-per-epoch', type=int, default=2500)
    parser.add_argument('--step-per-collect', type=int, default=1000)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--episode-per-collect', type=int, default=10)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--train-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=9)
    parser.add_argument('--episode-per-test', type=int, default=50)
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--logdir', type=str, default='log_tensorboard')
    parser.add_argument('--resume', action='store_true', default=False)
    # experiment special
    parser.add_argument('--model', type=str, default='paas')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--test-on-seen", action='store_true')
    parser.add_argument('--train-dataset', type=str, default='Wu2017')
    parser.add_argument('--test-dataset', type=str, default='Wu2017')
    parser.add_argument('--network-dataset', type=str, default='4G')
    parser.add_argument('--qoe-train-ids', type=int, nargs='*')  # train on one or more qoe
    parser.add_argument('--qoe-test-ids', type=int, nargs='*')  # test on one or more qoe
    parser.add_argument("--policy-path", type=str)
    args = parser.parse_known_args()[0]

    # debug
    # args.epochs = 10
    # args.step_per_epoch = 1000
    # args.step_per_collect = 500
    # args.batch_size = 2
    # args.train = False
    # args.test = True

    # python run_paas.py --epoch 100 --step-per-epoch 6000 --step-per-collect 2000 --batch-size 256 --train --train-dataset Wu2017 --test --test-dataset Wu2017 --test-on-seen --qoe-test-ids 0 --device cuda:1 --seed 1

    print(args)
    
    config = get_config_from_yml()
    run(args, config)
