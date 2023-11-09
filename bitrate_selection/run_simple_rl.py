import argparse
import os
import random
import numpy as np
import torch

from tqdm import tqdm
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import A2CPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic
from envs.simple_rl_env import SimpleRLEnv
from models.simple_rl import FeatureNet, Actor, Critic
from utils.common import get_config_from_yml, read_log_file


def train(args, config, policy, qoe_weights, models_dir, file_prefix):
    train_log_path = os.path.join(models_dir, file_prefix + '_train_log.csv')
    valid_log_path = os.path.join(models_dir, file_prefix + '_valid_log.csv')
    if os.path.exists(train_log_path):
        os.remove(train_log_path)
    if os.path.exists(valid_log_path):
        os.remove(valid_log_path)

    env = SimpleRLEnv(config, args.train_dataset, args.network_dataset, qoe_weights, valid_log_path, config.startup_download,
                  mode='valid', seed=args.seed, device=args.device)
    env.seed(args.seed)

    args.episode_per_test = env.sample_count()
    print('Training num:', args.train_num)
    print('Test num:', args.test_num)
    print('Episode per test:', args.episode_per_test)

    train_env = SubprocVectorEnv(
        [lambda: SimpleRLEnv(config, args.train_dataset, args.network_dataset, qoe_weights, train_log_path, config.startup_download,
                             mode='train', seed=args.seed, worker_num=args.train_num, device=args.device) for _ in range(args.train_num)],

    )
    valid_env = DummyVectorEnv(
        [lambda: SimpleRLEnv(config, args.train_dataset, args.network_dataset, qoe_weights, valid_log_path, config.startup_download,
                             mode='valid', seed=args.seed, worker_num=args.test_num, device=args.device) for _ in range(args.test_num)],
    )
    print('Training QoE weights:', qoe_weights)
    
    train_env.seed(args.seed)
    valid_env.seed(args.seed)
    
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

    replay_buffer = VectorReplayBuffer(
        total_size=args.buffer_size // len(train_env),
        buffer_num=len(train_env),
    )
    train_collector = Collector(
        policy=policy, env=train_env, buffer=replay_buffer
    )
    log_path = os.path.join(models_dir, file_prefix + '_rlva_tb_logger')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    valid_collector = Collector(policy, valid_env)

    trainer = OnpolicyTrainer(
        policy,
        train_collector,
        valid_collector,
        args.epochs,
        args.step_per_epoch,
        args.repeat_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        save_checkpoint_fn=save_checkpoint_fn,
    )
    
    for epoch, epoch_stat, info in trainer:
        print(f"Epoch: {epoch}")
        print(epoch_stat)
        print(info)


def test(args, config, policy, qoe_weights, models_dir, results_dir, file_prefix):
    test_log_path = os.path.join(results_dir, file_prefix + '_results.csv')
    if os.path.exists(test_log_path):
        os.remove(test_log_path)

    test_env = SimpleRLEnv(config, args.test_dataset, args.network_dataset, qoe_weights, test_log_path, config.startup_download,
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
            video, user, trace = test_env.current_video, test_env.current_user, test_env.current_trace
            done = False
            while not done:
                for key, value in state.items():
                    state[key] = np.expand_dims(value, 0)
                # batch = {'obs': state}
                batch = Batch(obs=state, info={})
                # logits = policy(batch, state=None).logits
                # action = F.softmax(logits, dim=-1).argmax().item()
                action = policy(batch, state=None).act
                action = action.item()
                state, reward, done, _ = test_env.step(action)
            state = test_env.reset()
        read_log_file(test_log_path)
        print('Results saved at:', test_log_path)


def run(args, config):
    assert args.qoe_train_id is not None

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    
    assert args.qoe_train_id is not None
    split = 'train' if args.test_on_seen else 'test'
    if args.qoe_test_ids is None:
        args.qoe_test_ids = list(range(len(config.qoe_split[split])))

    models_dir = os.path.join(config.bs_models_dir, args.model, args.train_dataset + '_' + args.network_dataset, f'qoe{args.qoe_train_id}')
    if args.test_on_seen:
        results_dir = os.path.join(config.bs_results_dir, args.model,  args.test_dataset + '_' + args.network_dataset, 'seen_qoe' + '_'.join(map(str, args.qoe_test_ids)))
    else:
        results_dir = os.path.join(config.bs_results_dir, args.model,  args.test_dataset + '_' + args.network_dataset, 'unseen_qoe' + '_'.join(map(str, args.qoe_test_ids)))
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    file_prefix = f'epochs_{args.epochs}_bs_{args.batch_size}_lr_{args.lr}_gamma_{args.gamma}_seed_{args.seed}_ent_{args.ent_coef}'

    feature_net = FeatureNet(config.past_k, config.tile_total_num, len(config.video_rates), device=args.device)
    actor = Actor(feature_net, feature_dim=5 * 128, action_space=config.action_space, device=args.device)
    critic = Critic(feature_net, feature_dim=5 * 128, device=args.device)
    model = ActorCritic(actor, critic)
    
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    def dist(logits):
        return Categorical(logits)
    
    policy = A2CPolicy(
        actor,
        critic,
        optimizer,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        action_space=config.action_space,
    ).to(args.device)

    torch.set_float32_matmul_precision('high')

    if args.train:
        qoe_weights = [config.qoe_split['train'][args.qoe_train_id]]
        print('Training QoE weights:', qoe_weights)
        train(args, config, policy, qoe_weights, models_dir, file_prefix)
    if args.test:
        qoe_weights = [config.qoe_split[split][i] for i in args.qoe_test_ids]
        print('Testing QoE weights:', qoe_weights)
        test(args, config, policy, qoe_weights, models_dir, results_dir, file_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='simple_rl')
    parser.add_argument('--reward-threshold', type=float, default=500000.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=2500)
    parser.add_argument('--step-per-collect', type=int, default=1000)
    parser.add_argument('--episode-per-collect', type=int, default=10)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--train-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=9)
    parser.add_argument('--episode-per-test', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--logdir', type=str, default='log_tensorboard')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument("--rew-norm", type=int, default=True)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.1)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bound-action-method", type=str, default="clip")
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    # experiment special
    parser.add_argument('--model', type=str, default='simple_rl')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--test-on-seen", action='store_true')
    parser.add_argument('--train-dataset', type=str, default='Jin2022')
    parser.add_argument('--test-dataset', type=str, default='Jin2022')
    parser.add_argument('--network-dataset', type=str, default='4G')
    parser.add_argument('--qoe-train-id', type=int)  # train on one qoe
    parser.add_argument('--qoe-test-ids', type=int, nargs='*')  # test on one or more qoe
    parser.add_argument("--policy-path", type=str)
    args = parser.parse_known_args()[0]


    # On Jin2022
    # python run_simple_rl.py --epochs 100 --step-per-epoch 6000 --step-per-collect 2000 --batch-size 256 --train --train-dataset Jin2022 --test --test-dataset Jin2022 --qoe-train-id 0 --qoe-test-ids 0 --test-on-seen --device cuda:0 --seed 1

    print(args)
    
    config = get_config_from_yml()
    run(args, config)
