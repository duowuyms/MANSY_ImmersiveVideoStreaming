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
from models.gen360 import Actor3, Critic3, QoEIdentifier3, FeatureNet3, QoEIdentifierFeatureNet3
from models.gen360_trainer import OnpolicyTrainer
from models.gen360_ppo import PPOPolicy
from utils.common import get_config_from_yml, read_log_file, normalize_qoe_weight
from utils.gen360_utils import train_identifier, behavior_cloning_pretraining
from utils.console_logger import ConsoleLogger


def case_study(args, config, policy, qoe_weights, identifier, results_dir, target_video=2, target_user=17, target_trace=23,
               qoe_weight1=[1, 7, 1], qoe_weight2=[7, 1, 1]):
    test_log_path = os.path.join(results_dir, 'results_ignore_this_file.csv')
    if os.path.exists(test_log_path):
        os.remove(test_log_path)

    test_env = GEN360Env(config, args.test_dataset, args.network_dataset, qoe_weights, identifier, args.lamb, test_log_path, config.startup_download,
                         mode='test', seed=args.seed, device=args.device)
    test_env.seed(args.seed)

    policy_path = args.policy_path
    assert policy_path is not None
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
        print("Successfully loaded agent from:", policy_path)
    else:
        raise FileExistsError(f"File not exist: {policy_path}")


    qoe_weight1 = np.array(qoe_weight1, dtype=np.float32)
    qoe_weight2 = np.array(qoe_weight2, dtype=np.float32)
    qoe, qoe1, qoe2, qoe3 = [], [], [], []
    with torch.no_grad():
        sample_count = test_env.sample_count()
        for i in tqdm(range(sample_count), desc='Testing: '):
            state = test_env.reset()
            video, user, trace = test_env.current_video, test_env.current_user, test_env.current_trace
            # print(video, user, trace, test_env.current_qoe_weight)
            if not (video == target_video and user == target_user and trace == target_trace):
                continue
            done = False
            chunk, total_chunk = 0, test_env.simulator.chunk_num
            while not done:
                if chunk <= total_chunk // 2:
                    state['qoe_weight'] = normalize_qoe_weight(qoe_weight1)
                else:
                    state['qoe_weight'] = normalize_qoe_weight(qoe_weight2)
                for key, value in state.items():
                    state[key] = np.expand_dims(value, 0)
                # batch = {'obs': state}
                batch = Batch(obs=state, info={})
                results = policy(batch, state=None)
                logits, act, dist = results.logits, results.act, results.dist
                action = act.item()
                state, reward, done, _ = test_env.step(action)
                try:
                    qoe.append(test_env.log_qoe[-1])
                    qoe1.append(test_env.log_qoe1[-1])
                    qoe2.append(test_env.log_qoe2[-1])
                    qoe3.append(test_env.log_qoe3[-1])
                except IndexError:
                    pass
                chunk += 1
            print()
    case_study_path = os.path.join(results_dir, f'results_case_study_uid_{args.use_identifier}_v_{target_video}_u_{target_user}_t{target_trace}_' \
                                   f'fqoe_{"_".join(map(str, list(map(int, qoe_weight1.tolist()))))}_sqoe_{"_".join(map(str, list(map(int, qoe_weight2.tolist()))))}.csv')
    with open(case_study_path, 'w') as file:
        file.write('qoe,quality,rebuffer,smoothness\n')
        for i in range(len(qoe)):
            file.write(f'{qoe[i]},{qoe1[i]},{qoe2[i]},{qoe3[i]}\n')
        file.close()
    print('Case study results save at:', case_study_path)

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

    # intitialize policy
    # feature_net = FeatureNet(config.past_k, config.tile_total_num, len(config.video_rates), args.hidden_dim, device=args.device)
    feature_dim = args.hidden_dim if args.use_lstm else args.hidden_dim * 10
    feature_net = FeatureNet3(config.past_k, config.tile_total_num, len(config.video_rates), args.hidden_dim, device=args.device, use_lstm=args.use_lstm)
    actor = Actor3(feature_net, feature_dim=feature_dim, hidden_dim=args.hidden_dim, action_space=config.action_space, device=args.device)
    critic = Critic3(feature_net, feature_dim=feature_dim, hidden_dim=args.hidden_dim, device=args.device)
    model = ActorCritic(actor, critic)
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize QoE identifier
    # identifier_feature_net = QoEIdentifierFeatureNet(config.past_k, config.tile_total_num, len(config.video_rates), config.action_space, args.hidden_dim, device=args.device)
    identifier_feature_net = QoEIdentifierFeatureNet3(config.past_k, config.tile_total_num, len(config.video_rates), config.action_space, args.hidden_dim, device=args.device, use_lstm=args.use_lstm)
    identifier = QoEIdentifier3(identifier_feature_net, feature_dim=feature_dim, hidden_dim=args.hidden_dim, device=args.device).to(args.device)
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
    
    qoe_weight1 = [1, 7, 1]
    qoe_weight2 = [7, 1, 1]
    # qoe_weight1 = [7, 1, 1]
    # qoe_weight2 = [1, 7, 1]

    qoe_weights = [config.qoe_split[split][i] for i in args.qoe_test_ids]
    args.policy_path = '/data/wuduo/2023_omnidirectional_vs/models/bitrate_selection/gen360_3/Wu2017_4G/qoe0_1_2_3/epochs_1000_bs_512_lr_0.0005_gamma_0.95_seed_5_ent_0.02_useid_False_lambda_0.5_ilr_0.0001_iur_2_bc_False/best_policy.pth'
    args.use_identifier = False
    prefix = f'epochs_{args.epochs}_bs_{args.batch_size}_lr_{args.lr}_gamma_{args.gamma}_seed_{args.seed}_ent_{args.ent_coef}_useid_{args.use_identifier}' \
            f'_lambda_{args.lamb}_ilr_{args.identifier_lr}_iur_{args.identifier_update_round}_bc_{args.bc or args.init_from_bc}'
    results_dir = os.path.join(config.bs_results_dir, args.model,  args.test_dataset + '_' + args.network_dataset, 'case_study', prefix)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print('No identifier, load agent from:', args.policy_path)
    case_study(args, config, policy, qoe_weights, identifier, results_dir, qoe_weight1=qoe_weight1, qoe_weight2=qoe_weight2)

    args.policy_path = '/data/wuduo/2023_omnidirectional_vs/models/bitrate_selection/gen360_3/Wu2017_4G/qoe0_1_2_3/epochs_1000_bs_512_lr_0.0005_gamma_0.95_seed_666_ent_0.02_useid_True_lambda_0.5_ilr_0.0001_iur_2_bc_True/best_policy.pth'
    args.use_identifier = True
    prefix = f'epochs_{args.epochs}_bs_{args.batch_size}_lr_{args.lr}_gamma_{args.gamma}_seed_{args.seed}_ent_{args.ent_coef}_useid_{args.use_identifier}' \
            f'_lambda_{args.lamb}_ilr_{args.identifier_lr}_iur_{args.identifier_update_round}_bc_{args.bc or args.init_from_bc}'
    results_dir = os.path.join(config.bs_results_dir, args.model,  args.test_dataset + '_' + args.network_dataset, 'case_study', prefix)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print('Use identifier, load agent from:', args.policy_path)
    case_study(args, config, policy, qoe_weights, identifier, results_dir, qoe_weight1=qoe_weight1, qoe_weight2=qoe_weight2)


if __name__ == '__main__':
    # python run_PAAS.py --seed 1 --lr 0.0001 --gamma 0.99 --batch-size 256 --device cuda:0 --model paas1 --mode train
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='gen360_3')
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
    parser.add_argument('--model', type=str, default='gen360_3')
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
    args.epochs = 2000
    args.identifier_epochs = 1000
    args.step_per_epoch = 600
    args.step_per_collect = 200
    args.batch_size = 512
    args.qoe_train_ids = [0, 1]
    args.qoe_test_ids = [3]
    # args.policy_path = '/data/wuduo/2023_omnidirectional_vs/models/bitrate_selection/gen360/Wu2017_4G/qoe0_1_2_3/epochs_2000_bs_512_lr_0.0001_gamma_0.95_seed_1_ent_0.02_useid_True_lambda_0.5_ilr_0.0001_iur_1_bc_True/best_policy.pth'
    # no repl
    # args.policy_path = '/data/wuduo/2023_omnidirectional_vs/models/bitrate_selection/gen360_3/Wu2017_4G/qoe0_1_2_3/epochs_1000_bs_512_lr_0.0005_gamma_0.95_seed_5_ent_0.02_useid_False_lambda_0.5_ilr_0.0001_iur_2_bc_False/best_policy.pth'
    # with repl
    # args.policy_path = '/data/wuduo/2023_omnidirectional_vs/models/bitrate_selection/gen360_3/Wu2017_4G/qoe0_1_2_3/epochs_1000_bs_512_lr_0.0005_gamma_0.95_seed_666_ent_0.02_useid_True_lambda_0.5_ilr_0.0001_iur_2_bc_True/best_policy.pth'
    # args.bc = True
    # args.bc_max_steps = 10
    # args.bc_identifier_max_steps = 10
    # args.init_from_bc = True

    # No BC
    # python run_gen360.py --epoch 200 --step-per-epoch 6000 --step-per-collect 2000 --lr 0.0001 --batch-size 256 --train --train-dataset Wu2017 --test --test-dataset Wu2017 --qoe-test-ids 0 --test-on-seen --lamb 0.5 --train-identifier --identifier-epoch 50 --device cuda:3
    # test (only)
    # python run_gen360.py --epoch 150 --step-per-epoch 6000 --step-per-collect 2000 --lr 0.0001 --batch-size 256 --train-dataset Jin2022 --test --test-dataset Jin2022 --qoe-test-ids 1 --test-on-seen --lamb 0.5 --device cuda:3

    # BC
    # python run_gen360.py --epoch 1000 --step-per-epoch 4096 --step-per-collect 2048 --lr 0.0001 --batch-size 512 --train --train-dataset Wu2017 --test --test-dataset Wu2017 --qoe-test-ids 0 1 2 3 --test-on-seen --lamb 0.5 --train-identifier --identifier-epoch 700 --bc --bc-max-steps 500 --bc-valid-per-step 50 --bc-identifier-max-steps 500 --device cuda:1 --gamma 0.95 --ent-coef 0.02 --seed 5 --use-identifier

    print(args)
    
    config = get_config_from_yml()
    run(args, config)
