from typing import Any, Callable, DefaultDict, Dict, Optional, Tuple, Union

import numpy as np
import tqdm

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer import BaseTrainer as TianshouBaseTrainer
from tianshou.trainer.utils import gather_info
from tianshou.utils import BaseLogger, LazyLogger, tqdm_config
from utils.mansy_utils import train_identifier


class BaseTrainer(TianshouBaseTrainer):
    """
    A customized basetrainer that supports our representation learning based training.
    """
    def __next__(self) -> Union[None, Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        """Perform one epoch (both train and eval)."""
        self.epoch += 1
        self.iter_num += 1

        if self.iter_num > 1:

            # iterator exhaustion check
            if self.epoch >= self.max_epoch:
                raise StopIteration

            # exit flag 1, when stop_fn succeeds in train_step or test_step
            if self.stop_fn_flag:
                raise StopIteration

        # set policy in train mode
        self.policy.train()

        epoch_stat: Dict[str, Any] = dict()
        # perform n step_per_epoch
        with tqdm.tqdm(
            total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", **tqdm_config
        ) as t:
            while t.n < t.total and not self.stop_fn_flag:
                data: Dict[str, Any] = dict()
                result: Dict[str, Any] = dict()
                if self.train_collector is not None:
                    data, result, self.stop_fn_flag = self.train_step()
                    t.update(result["n/st"])
                    if self.stop_fn_flag:
                        t.set_postfix(**data)
                        break
                else:
                    assert self.buffer, "No train_collector or buffer specified"
                    result["n/ep"] = len(self.buffer)
                    result["n/st"] = int(self.gradient_step)
                    t.update()

                if self.args.train_identifier:
                    print('==================== Start Training QOE identifier ====================')
                    train_identifier(self.identifier,  self.identifier_optimizer, tracjetory=self.train_collector.buffer, update_round=self.args.identifier_update_round)
                    print('==================== End Training identifier ====================')

                self.policy_update_fn(data, result, is_train=True)
                t.set_postfix(**data)

            if t.n <= t.total and not self.stop_fn_flag:
                t.update()

        if not self.stop_fn_flag:
            self.logger.save_data(
                self.epoch, self.env_step, self.gradient_step, self.save_checkpoint_fn
            )
            # test
            if self.test_collector is not None:
                test_stat, self.stop_fn_flag = self.test_step()
                if not self.is_run:
                    epoch_stat.update(test_stat)

        if not self.is_run:
            epoch_stat.update({k: v.get() for k, v in self.stat.items()})
            epoch_stat["gradient_step"] = self.gradient_step
            epoch_stat.update(
                {
                    "env_step": self.env_step,
                    "rew": self.last_rew,
                    "len": int(self.last_len),
                    "n/ep": int(result["n/ep"]),
                    "n/st": int(result["n/st"]),
                }
            )
            info = gather_info(
                self.start_time, self.train_collector, self.test_collector,
                self.best_reward, self.best_reward_std
            )
            return self.epoch, epoch_stat, info
        else:
            return None


class OnpolicyTrainer(BaseTrainer):
    """
    A customized OnpolicyTrainer that supports our representation learning based training.
    Mostly it is simimlar to the tianshou.trainer.OnpolicyTrainer.
    """
    __doc__ = BaseTrainer.gen_doc("onpolicy") + "\n".join(__doc__.split("\n")[1:])

    def __init__(
        self,
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Optional[Collector],
        max_epoch: int,
        step_per_epoch: int,
        repeat_per_collect: int,
        episode_per_test: int,
        batch_size: int,
        step_per_collect: Optional[int] = None,
        episode_per_collect: Optional[int] = None,
        train_fn: Optional[Callable[[int, int], None]] = None,
        test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_best_fn: Optional[Callable[[BasePolicy], None]] = None,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
        resume_from_log: bool = False,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        logger: BaseLogger = LazyLogger(),
        verbose: bool = True,
        test_in_train: bool = True,
        args = None,
        identifier = None,
        identifier_optimizer = None,
        **kwargs: Any,
    ):
        super().__init__(
            learning_type="onpolicy",
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=max_epoch,
            step_per_epoch=step_per_epoch,
            repeat_per_collect=repeat_per_collect,
            episode_per_test=episode_per_test,
            batch_size=batch_size,
            step_per_collect=step_per_collect,
            episode_per_collect=episode_per_collect,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            resume_from_log=resume_from_log,
            reward_metric=reward_metric,
            logger=logger,
            verbose=verbose,
            test_in_train=test_in_train,
            **kwargs,
        )
        self.args = args
        self.identifier = identifier
        self.identifier_optimizer = identifier_optimizer
        self.cnt = 0
        self.observe_round = 1000

    def policy_update_fn(
        self, data: Dict[str, Any], result: Optional[Dict[str, Any]] = None, is_train=False
    ) -> None:
        """Perform one on-policy update."""
        assert self.train_collector is not None
        losses = self.policy.update(
            0,
            self.train_collector.buffer,
            batch_size=self.batch_size,
            repeat=self.repeat_per_collect,
            is_train=is_train
        )
        self.train_collector.reset_buffer(keep_statistics=True)
        step = max([1] + [len(v) for v in losses.values() if isinstance(v, list)])
        self.gradient_step += step
        self.log_update_data(data, losses)


def onpolicy_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for OnpolicyTrainer run method.

    It is identical to ``OnpolicyTrainer(...).run()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return OnpolicyTrainer(*args, **kwargs).run()


onpolicy_trainer_iter = OnpolicyTrainer
