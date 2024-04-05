from typing import Optional, Union

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline

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
        policy_kwargs: dict = {},
        baseline_kwargs: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = DeepACOPlusPolicy(env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
