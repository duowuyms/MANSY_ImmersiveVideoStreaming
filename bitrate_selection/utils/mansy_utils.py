import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import choice
from copy import deepcopy


def train_identifier(identifier, identifier_optim, tracjetory, update_round=2):
    """
    Train QoE identifier over the given trajectory.
    """
    samples, _ = tracjetory.sample(0)
    samples = deepcopy(samples)
    identifier.train()
    train_ratio = 0.8
    num_samples = len(samples)
    idx = np.arange(num_samples)
    np.random.shuffle(idx)
    samples = samples[idx]
    num_train = int(num_samples * train_ratio)
    train_obs = samples['obs'][:num_train]
    valid_obs = samples['obs'][num_train:]
    for _ in range(update_round):
        pred_qoe_weight = identifier(train_obs, train_obs['action_one_hot'])
        gt_qoe_weight = torch.from_numpy(train_obs['qoe_weight']).to(identifier.device)

        loss = F.mse_loss(pred_qoe_weight, gt_qoe_weight)
        identifier_optim.zero_grad()
        loss.backward()
        identifier_optim.step()
        print('identifier loss is: ', loss.item())
    
    identifier.eval()
    with torch.no_grad():
        pred_qoe_weight = identifier(valid_obs, valid_obs['action_one_hot'])
        gt_qoe_weight = torch.from_numpy(valid_obs['qoe_weight']).to(identifier.device)
        valid_loss = F.mse_loss(pred_qoe_weight, gt_qoe_weight)
        print('identifier validation loss is: ', valid_loss.item())


def calculate_indentifier_reward(identifier, state, action_one_hot):
    identifier.eval()
    with torch.no_grad():
        pred_qoe_weight = identifier(state, action_one_hot)
        gt_qoe_weight = torch.from_numpy(state['qoe_weight']).to(identifier.device).reshape((1, 3))
        mse = F.mse_loss(pred_qoe_weight, gt_qoe_weight)
    identifier_reward = 1 - mse
    return identifier_reward.cpu().numpy()


def behavior_cloning_pretraining(args, policy, identifier, policy_optim, identifier_optim, train_demos, valid_demos, 
                                 max_steps, valid_per_step, identifier_max_steps, identifier_update_round,
                                 policy_save_path, identifier_save_path):
    """
    Behavior cloning initialization.
    This is a common trick in training the RL models.
    However, we find that this trick does not lead to any noticable improvement (e.g., faster convergence or improved reward) in our case,
    so we do not report it in our paper.
    """
    criterion = nn.CrossEntropyLoss()
    best_loss, best_step = float('inf'), 0
    bs = 32
    for i in range(max_steps):
        demo = choice(train_demos)
        samples, _ = demo.sample(0)
        batch = policy(samples)
        logits, act, dist = batch.logits, batch.act, batch.dist
        loss = criterion(logits, torch.tensor(samples['act']).to(args.device)) - 0.1 * dist.entropy().mean()
        policy_optim.zero_grad()
        loss.backward()
        policy_optim.step()
        print(f'BC (Training): loss={loss.item()} ({i + 1}/{max_steps})')

        if i % valid_per_step == 0:
            policy.eval()
            valid_loss = 0.
            for j in range(len(valid_demos)):
                valid_samples, _ = valid_demos[j].sample(0)
                logits = policy(valid_samples).logits
                loss = criterion(logits, torch.tensor(valid_samples['act']).to(args.device))
                valid_loss += loss.item()
            valid_loss = valid_loss / len(valid_demos)
            if best_loss > valid_loss:
                best_loss = valid_loss
                best_step = i
                torch.save(policy.state_dict(), policy_save_path)
            print(f'BC (Validation): valid loss={valid_loss} - best loss={best_loss} at step {best_step}')
            policy.train()
        
        if i < identifier_max_steps:
            train_identifier(identifier, identifier_optim, demo, identifier_update_round)
            torch.save(identifier.state_dict(), identifier_save_path)

