import sys
import argparse
import os
import random
import pickle
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.distributions import Categorical, Independent
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Batch, Collector, VectorReplayBuffer, ReplayBuffer, AsyncCollector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv, RayVectorEnv
# from tianshou.policy import PPOPolicy
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic
from envs.gen360_env import GEN360Env
from models.gen360 import FeatureNet, Actor, Critic, QoEIdentifier, QoEIdentifierFeatureNet, FeatureNet2, QoEIdentifierFeatureNet2
from models.gen360_trainer import OnpolicyTrainer
from models.gen360_ppo import PPOPolicy
from utils.common import get_config_from_yml, read_log_file
from utils.gen360_utils import train_identifier, behavior_cloning_pretraining
from utils.console_logger import ConsoleLogger


def train(args, config, policy, qoe_weights, identifier, identifier_optimizer, models_dir, policy_bc_path, identifier_bc_path):
    train_log_path = os.path.join(models_dir, 'train_log.csv')
    valid_log_path = os.path.join(models_dir, 'valid_log.csv')
    if os.path.exists(train_log_path):
        os.remove(train_log_path)
    if os.path.exists(valid_log_path):
        os.remove(valid_log_path)

    env = GEN360Env(config, args.train_dataset, args.network_dataset, qoe_weights, identifier, args.lamb, valid_log_path, config.startup_download,
                    mode='valid', seed=args.seed, device=args.device, use_identifier=args.use_identifier)
    env.seed(args.seed)

    args.train_num = 1
    args.test_num = len(qoe_weights)
    args.episode_per_test = env.sample_count()
    print('Training num:', args.train_num)
    print('Test num:', args.test_num)
    print('Episode per test:', args.episode_per_test)

    train_env = DummyVectorEnv(
        [lambda: GEN360Env(config, args.train_dataset, args.network_dataset, qoe_weights, identifier, args.lamb, train_log_path, config.startup_download,
                           mode='train', seed=args.seed, worker_num=args.train_num, device=args.device, use_identifier=args.use_identifier) for _ in range(args.train_num)],

    )
    valid_env = DummyVectorEnv(
        [lambda: GEN360Env(config, args.train_dataset, args.network_dataset, qoe_weights, identifier, args.lamb, valid_log_path, config.startup_download,
                           mode='valid', seed=args.seed, worker_num=args.test_num, device=args.device, use_identifier=args.use_identifier) for _ in range(args.test_num)],
    )
    print('Training QoE weights:', qoe_weights)
    
    train_env.seed(args.seed)
    valid_env.seed(args.seed)
    
    checkpoint_path = os.path.join(models_dir, 'checkpoint.pth')
    identifier_checkpoint_path = os.path.join(models_dir, 'identifier_checkpoint.pth')
    best_policy_path = os.path.join(models_dir, 'best_policy.pth')
    best_identifier_path = os.path.join(models_dir, 'best_identifier.pth')
    if args.resume:
        if os.path.exists(checkpoint_path):
            policy.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
            print("Successfully loaded agent from:", checkpoint_path)
        else:
            print('Failed to load agent:', checkpoint_path, 'no such file')
        if os.path.exists(identifier_checkpoint_path):
            identifier.load_state_dict(torch.load(identifier_checkpoint_path, map_location=args.device))
            print("Successfully init identifier from behavior cloning:", identifier_checkpoint_path)
        else:
            print('Failed to load identifier:', identifier_checkpoint_path, 'no such file')
    elif args.init_from_bc:
        if os.path.exists(policy_bc_path):
            policy.load_state_dict(torch.load(policy_bc_path, map_location=args.device))
            print("Successfully init agent from behavior cloning:", policy_bc_path)
        else:
            print('Failed to load agent:', policy_bc_path, 'no such file')
        if os.path.exists(identifier_bc_path):
            identifier.load_state_dict(torch.load(identifier_bc_path, map_location=args.device))
            print("Successfully init identifier from behavior cloning:", identifier_bc_path)
        else:
            print('Failed to load identifier:', identifier_bc_path, 'no such file')
    
    def save_best_fn(policy):
        torch.save(policy.state_dict(), best_policy_path)
        torch.save(identifier.state_dict(), best_identifier_path)
        print('====================================================================')
        print('Best policy save at ' + best_policy_path)
        print('Best identifier save at ' + best_identifier_path)
        print('====================================================================')

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(policy.state_dict(), checkpoint_path)
        torch.save(identifier.state_dict(), identifier_checkpoint_path)
        print('====================================================================')
        print('Checkpoint saved at ' + checkpoint_path)
        print('Identifier checkpoint saved at ' + identifier_checkpoint_path)
        print('====================================================================')
        return checkpoint_path

    # policy.set_eps(0.1)

    # collector
    # replay_buffer = ReplayBuffer(args.buffer_size)
    replay_buffer = VectorReplayBuffer(
        total_size=args.buffer_size // len(train_env),
        buffer_num=len(train_env),
    )
    train_collector = Collector(
        policy=policy, env=train_env, buffer=replay_buffer
    )
    log_path = os.path.join(models_dir, 'gen360_tb_logger')
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
        args=args,
        identifier=identifier,
        identifier_optimizer=identifier_optimizer,
    )
    
    for epoch, epoch_stat, info in trainer:
        print(f"Epoch: {epoch}")
        # print(epoch_stat)
        # print(info)
        print('loss:', epoch_stat['loss'], ' --- ', 'loss/clip:', epoch_stat['loss/clip'], ' --- ', 'loss/vf:', epoch_stat['loss/vf'], ' --- ', 'loss/ent:', epoch_stat['loss/ent'])
        # if epoch_stat['loss/ent'] < 1.5:
        #     return

        # train QoE identifier
        # if args.train_identifier:
        #     print('==================== Start Training QOE identifier ====================')
        #     train_identifier(identifier,  identifier_optimizer, tracjetory=replay_buffer, update_round=args.identifier_update_round)
        #     print('==================== End Training identifier ====================')
        #     if epoch >= args.identifier_epochs:
        #         args.train_identifier = False

        # replay_buffer.reset()


def test(args, config, policy, qoe_weights, identifier, models_dir, results_dir):
    test_log_path = os.path.join(results_dir, 'results.csv')
    if os.path.exists(test_log_path):
        os.remove(test_log_path)

    test_env = GEN360Env(config, args.test_dataset, args.network_dataset, qoe_weights, identifier, args.lamb, test_log_path, config.startup_download,
                         mode='test', seed=args.seed, device=args.device)
    test_env.seed(args.seed)

    policy_path = args.policy_path
    if policy_path is None:
        policy_path = os.path.join(models_dir, 'best_policy.pth')
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
        print("Successfully loaded agent from:", policy_path)
    else:
        # raise FileExistsError(f"File not exist: {policy_path}")
        pass

    with torch.no_grad():
        state = test_env.reset()
        sample_count = test_env.sample_count()
        for i in tqdm(range(sample_count), desc='Testing: '):
            video, user, trace = test_env.current_video, test_env.current_user, test_env.current_trace
            # print(video, user, trace, test_env.current_qoe_weight)
            done = False
            while not done:
                for key, value in state.items():
                    state[key] = np.expand_dims(value, 0)
                # batch = {'obs': state}
                batch = Batch(obs=state, info={})
                results = policy(batch, state=None)
                logits, act, dist = results.logits, results.act, results.dist
                # action = F.softmax(logits, dim=-1).argmax().item()
                action = act.item()
                # action = Categorical(logits).sample().item()
                state, reward, done, _ = test_env.step(action)
                # if i < 40:
                #     print(action, end=',')
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
    
    prefix = f'epochs_{args.epochs}_bs_{args.batch_size}_lr_{args.lr}_gamma_{args.gamma}_seed_{args.seed}_ent_{args.ent_coef}_useid_{args.use_identifier}' \
             f'_lambda_{args.lamb}_ilr_{args.identifier_lr}_iur_{args.identifier_update_round}_bc_{args.bc or args.init_from_bc}'
    models_dir = os.path.join(config.bs_models_dir, args.model, args.train_dataset + '_' + args.network_dataset, 'qoe' + '_'.join(map(str, args.qoe_train_ids)), prefix)
    if args.test_on_seen:
        results_dir = os.path.join(config.bs_results_dir, args.model,  args.test_dataset + '_' + args.network_dataset, 'seen_qoe' + '_'.join(map(str, args.qoe_test_ids)), prefix)
    else:
        results_dir = os.path.join(config.bs_results_dir, args.model,  args.test_dataset + '_' + args.network_dataset, 'unseen_qoe' + '_'.join(map(str, args.qoe_test_ids)), prefix)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # intitialize policy
    # feature_net = FeatureNet(config.past_k, config.tile_total_num, len(config.video_rates), args.hidden_dim, device=args.device)
    feature_dim = args.hidden_dim if args.use_lstm else args.hidden_dim * 10
    feature_net = FeatureNet2(config.past_k, config.tile_total_num, len(config.video_rates), args.hidden_dim, device=args.device, use_lstm=args.use_lstm)
    actor = Actor(feature_net, feature_dim=feature_dim, hidden_dim=args.hidden_dim, action_space=config.action_space, device=args.device)
    critic = Critic(feature_net, feature_dim=feature_dim, hidden_dim=args.hidden_dim, device=args.device)
    model = ActorCritic(actor, critic)
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize QoE identifier
    # identifier_feature_net = QoEIdentifierFeatureNet(config.past_k, config.tile_total_num, len(config.video_rates), config.action_space, args.hidden_dim, device=args.device)
    identifier_feature_net = QoEIdentifierFeatureNet2(config.past_k, config.tile_total_num, len(config.video_rates), config.action_space, args.hidden_dim, device=args.device, use_lstm=args.use_lstm)
    identifier = QoEIdentifier(identifier_feature_net, feature_dim=feature_dim, hidden_dim=args.hidden_dim, device=args.device).to(args.device)
    for m in identifier.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    identifier_optimizer = torch.optim.Adam(identifier.parameters(), lr=args.identifier_lr, weight_decay=args.weight_decay)

    def dist(logits):
        return Categorical(logits=logits)

    policy = PPOPolicy(
        actor,
        critic,
        optimizer,
        dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        gae_lambda=args.gae_lambda,
        action_space=config.action_space,
        action_scaling=False,
        args=args,
        identifier=identifier
    ).to(args.device)

    torch.set_float32_matmul_precision('high')
    if args.train:
        console_log = open(os.path.join(models_dir, 'console.log'), 'w')
        sys.stdout = ConsoleLogger(sys.__stdout__, console_log)
        bc_file_prefix = f'bc_ms_{args.bc_max_steps}_ims_{args.bc_identifier_max_steps}_ilr_{args.identifier_lr}_iur_{args.identifier_update_round}'
        policy_bc_path = os.path.join(models_dir, bc_file_prefix + '_policy.pth')
        identifier_bc_path = os.path.join(models_dir, bc_file_prefix + '_identifier.pth')
        if args.bc:  # behavior cloning pretraining
            # fetch expert demonstrations
            demos_dir = os.path.join(config.bs_models_dir, 'expert', args.train_dataset + '_' + args.network_dataset, 'qoe' + '_'.join(map(str, args.qoe_train_ids)))
            train_demos_path = os.path.join(demos_dir, 'train_demonstrations.pkl')
            valid_demos_path = os.path.join(demos_dir, 'valid_demonstrations.pkl')
            assert os.path.exists(train_demos_path) and os.path.exists(valid_demos_path)
            train_demos = pickle.load(open(train_demos_path, 'rb'))
            train_demos = list(train_demos.values())  # demos are originally stored as {id: demo} dict
            valid_demos = pickle.load(open(valid_demos_path, 'rb'))
            valid_demos = list(valid_demos.values()) 

            behavior_cloning_pretraining(args, policy, identifier, optimizer, identifier_optimizer, train_demos, valid_demos, max_steps=args.bc_max_steps, 
                                        valid_per_step=args.bc_valid_per_step, identifier_max_steps=args.bc_identifier_max_steps, 
                                        identifier_update_round=args.identifier_update_round, policy_save_path=policy_bc_path,
                                        identifier_save_path=identifier_bc_path)
        qoe_weights = [config.qoe_split['train'][i] for i in args.qoe_train_ids]
        train(args, config, policy, qoe_weights, identifier, identifier_optimizer, models_dir, policy_bc_path, identifier_bc_path)
    if args.test:
        qoe_weights = [config.qoe_split[split][i] for i in args.qoe_test_ids]
        test(args, config, policy, qoe_weights, identifier, models_dir, results_dir)


if __name__ == '__main__':
    # python run_PAAS.py --seed 1 --lr 0.0001 --gamma 0.99 --batch-size 256 --device cuda:0 --model paas1 --mode train
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='gen360_wo_residual')
    parser.add_argument('--reward-threshold', type=float, default=500000.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
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
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=1)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)  # 3.0
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--save-interval", type=int, default=4)
    # experiment special
    parser.add_argument('--model', type=str, default='gen360_wo_residual')
    parser.add_argument('--use-lstm', action='store_true')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--identifier-lr', type=float, default=1e-4)
    parser.add_argument('--identifier-update-round', type=int, default=2)
    parser.add_argument('--identifier-epochs', type=int, default=50)
    parser.add_argument('--lamb', type=float, default=0.5)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--train-identifier", action='store_true')
    parser.add_argument("--use-identifier", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--test-on-seen", action='store_true')
    parser.add_argument('--train-dataset', type=str, default='Wu2017')
    parser.add_argument('--test-dataset', type=str, default='Wu2017')
    parser.add_argument('--network-dataset', type=str, default='4G')
    parser.add_argument('--qoe-train-ids', type=int)  # train on one qoe
    parser.add_argument('--qoe-test-ids', type=int, nargs='*')  # test on one or more qoe
    parser.add_argument("--policy-path", type=str)
    # behavior cloning special
    parser.add_argument("--bc", action='store_true')
    parser.add_argument("--bc-max-steps", type=int, default=500)
    parser.add_argument("--bc-valid-per-step", type=int, default=50)
    parser.add_argument("--bc-identifier-max-steps", type=int, default=100)
    parser.add_argument("--init-from-bc", action='store_true')


    args = parser.parse_known_args()[0]

    # debug
    # args.epochs = 20
    # args.identifier_epochs = 20
    # args.step_per_epoch = 600
    # args.step_per_collect = 200
    # args.batch_size = 256
    # args.train = False
    # args.test = True
    # args.train_identifier = True
    # args.use_identifier = True
    # args.qoe_train_ids = [0, 1]
    # args.qoe_test_ids = [0, 1, 2, 3]
    # args.policy_path = '/data/wuduo/2023_omnidirectional_vs/models/bitrate_selection/gen360/Wu2017_4G/qoe0_1_2_3/epochs_2000_bs_512_lr_0.0001_gamma_0.95_seed_1_ent_0.02_useid_True_lambda_0.5_ilr_0.0001_iur_1_bc_True/best_policy.pth'
    # args.bc = True
    # args.bc_max_steps = 10
    # args.bc_identifier_max_steps = 10
    # args.init_from_bc = True

    # No BC
    # python run_gen360.py --epoch 1000 --step-per-epoch 5000 --step-per-collect 2500 --lr 0.0005 --batch-size 512 --train --train-dataset Wu2017 --test --test-dataset Wu2017 --qoe-test-ids 0 1 2 3 --test-on-seen --lamb 0.5 --train-identifier --identifier-epoch 1000  --device cuda:0 --gamma 0.95 --ent-coef 0.02 --seed 666 --use-identifier
    # test (only)
    # python run_gen360.py --epoch 150 --step-per-epoch 6000 --step-per-collect 2000 --lr 0.0001 --batch-size 256 --train-dataset Jin2022 --test --test-dataset Jin2022 --qoe-test-ids 1 --test-on-seen --lamb 0.5 --device cuda:3

    # BC
    # python run_gen360.py --epoch 1000 --step-per-epoch 4096 --step-per-collect 2048 --lr 0.0001 --batch-size 512 --train --train-dataset Wu2017 --test --test-dataset Wu2017 --qoe-test-ids 0 1 2 3 --test-on-seen --lamb 0.5 --train-identifier --identifier-epoch 700 --bc --bc-max-steps 500 --bc-valid-per-step 50 --bc-identifier-max-steps 500 --device cuda:1 --gamma 0.95 --ent-coef 0.02 --seed 5 --use-identifier

    print(args)
    
    config = get_config_from_yml()
    run(args, config)
