from typing import Optional

import torch

from torch import Tensor

from rl4co.models.zoo.deepaco.antsystem import AntSystem


class ElitistAS(AntSystem):
    def __init__(
        self,
        log_heuristic: Tensor,
        n_ants: int = 20,
        alpha: float = 1,
        beta: float = 1,
        decay: float = 0.95,
        pheromone: Optional[Tensor] = None,
        require_logp: bool = False,
        use_local_search: bool = False,
        local_search_params: dict = ...,
    ):
        super().__init__(
            log_heuristic,
            n_ants,
            alpha,
            beta,
            decay,
            pheromone,
            require_logp,
            use_local_search,
            local_search_params,
        )
        self.best_index = None

    def _update_results(self, actions, reward):
        best_index = super()._update_results(actions=actions, reward=reward)
        self.best_index = best_index
        return best_index

    def _update_pheromone(self, actions, reward):
        # calculate Δphe
        delta_pheromone = torch.zeros_like(self.pheromone)
        from_node = actions[self._batchindex, self.best_index, None]
        to_node = torch.roll(from_node, 1, -1)
        mapped_reward = self._reward_map(
            reward[self._batchindex, self.best_index, None]
        ).detach()
        batch_action_indices = self._batch_action_indices(
            self.batch_size, actions.shape[-1], reward.device
        )

        ant_index = 0
        delta_pheromone[
            batch_action_indices,
            from_node[:, ant_index].flatten(),
            to_node[:, ant_index].flatten(),
        ] += mapped_reward[batch_action_indices, ant_index]

        # decay & update
        self.pheromone *= self.decay
        self.pheromone += delta_pheromone
