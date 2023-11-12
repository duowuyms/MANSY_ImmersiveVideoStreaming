from typing import Any, Dict, List, Optional, Type

import torch

from tianshou.data import ReplayBuffer
from tianshou.policy import PPOPolicy as TianshouPPOPolicy
from utils.mansy_utils import calculate_indentifier_reward


class PPOPolicy(TianshouPPOPolicy):
    """
    A customized PPOPolicy that supports our representation learning based training.
    """
    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        args = None,
        identifier = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, eps_clip, dual_clip, value_clip,
                         advantage_normalization, recompute_advantage, **kwargs)
        self.args = args
        self.identifier = identifier
        self.cnt = 0
        self.observe_round = 1000    

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer], is_train=False,
               **kwargs: Any) -> Dict[str, Any]:
        if buffer is None:
            return {}
        batch, indices = buffer.sample(sample_size)
        if self.args.use_identifier and is_train:
            # recompute reward
            for i in range(len(batch)):
                obs = batch.obs[i]
                action_one_hot = obs.action_one_hot
                qoe_reward = batch.rew[i]
                identifier_reward = calculate_indentifier_reward(self.identifier, obs, action_one_hot)
                batch.rew[i] = (1 - self.args.lamb) * qoe_reward + self.args.lamb * identifier_reward
                self.cnt += 1
                if self.cnt % self.observe_round == 0:
                    print('Reward:', batch.rew[i], ' --- ', 'QoE Reward:', qoe_reward, ' --- ', 'Identifier Reward:', identifier_reward)
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)
        result = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False
        return result
