from typing import Any, Optional, Union

import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.utils.utils import merge_with_defaults

from .policy import DeepACOPlusPolicy


class DeepACOPlus(REINFORCE):
    """
    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline. Defaults to exponential
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[DeepACOPlusPolicy] = None,
        baseline: Union[REINFORCEBaseline, str] = "exponential",
        train_kwargs: Optional[dict] = None,
        policy_kwargs: dict = {},
        baseline_kwargs: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = DeepACOPlusPolicy(env.name, **policy_kwargs)

        self.train_kwargs = merge_with_defaults(train_kwargs, gamma=1.0)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
        out = self.policy(td, self.env, phase=phase)

        # Compute loss
        if phase == "train":
            out = self.calculate_loss(td, batch, out)
            reward_loss = out.get("loss", None)
            unique_loss = self.unique_loss(td, batch, out)
            loss = reward_loss + self.train_kwargs["gamma"] * unique_loss

            out.update(
                dict(
                    loss_rw=reward_loss,
                    loss_u=unique_loss,
                    loss=loss,
                )
            )
            metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
            return {
                "loss_rw": reward_loss,
                "loss_u": unique_loss,
                "loss": loss,
                **metrics,
            }
        else:
            metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
            return {"loss": None, "loss_rw": None, "loss_u": None, **metrics}

    @staticmethod
    def unique_loss(td, batch, out):
        heatmap_logp = out["heatmap"]
        heatmap = nn.functional.softmax(heatmap_logp, -1)
        colsum = heatmap.sum(-2)
        loss_u = ((colsum - 1) ** 2).sum(-1).mean()
        return loss_u
