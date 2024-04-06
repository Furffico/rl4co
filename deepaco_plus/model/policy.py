from typing import Optional, Union

import torch.nn as nn

from tensordict import TensorDict

from rl4co.envs import get_env
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.nn.utils import get_log_likelihood
from rl4co.models.zoo.common.nonautoregressive.policy import NonAutoregressivePolicy
from rl4co.utils.pylogger import get_pylogger

from .decoder import DeepACOPlusDecoder

log = get_pylogger(__name__)


class DeepACOPlusPolicy(NonAutoregressivePolicy):
    """Implememts DeepACO policy based on :class:`NonAutoregressivePolicy`.

    Args:
        env_name: Name of the environment used to initialize embeddings
        encoder: Encoder module. Can be passed by sub-classes
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        edge_embedding: Model to use for the edge embedding. If None, use the default embedding for the environment
        heatmap_generator: Model to use for converting the edge embeddings to the heuristic information.
            If None, use the default MLP defined in :class:`~rl4co.models.zoo.common.nonautoregressive.decoder.EdgeHeatmapGenerator`.
        embedding_dim: Dimension of the embeddings
        num_encoder_layers: Number of layers in the encoder
        num_decoder_layers: Number of layers in the decoder
        **decoder_kwargs: Additional arguments to be passed to the DeepACO decoder.
    """

    def __init__(
        self,
        env_name: Union[str, RL4COEnvBase] = "tsp",
        encoder: Optional[nn.Module] = None,
        init_embedding: Optional[nn.Module] = None,
        edge_embedding: Optional[nn.Module] = None,
        heatmap_generator: Optional[nn.Module] = None,
        embedding_dim: int = 64,
        num_encoder_layers: int = 15,
        num_decoder_layers: int = 5,
        **decoder_kwargs,
    ):
        env_name_: str = env_name.name if isinstance(env_name, RL4COEnvBase) else env_name

        decoder = DeepACOPlusDecoder(
            env_name=env_name_,
            embedding_dim=embedding_dim,
            num_layers=num_decoder_layers,
            heatmap_generator=heatmap_generator,
            **decoder_kwargs,
        )

        super(DeepACOPlusPolicy, self).__init__(
            env_name,
            encoder,
            decoder,
            init_embedding,
            edge_embedding,
            embedding_dim,
            num_encoder_layers,
            num_decoder_layers,
            train_decode_type="multistart_sampling",
            val_decode_type="multistart_sampling",
            test_decode_type="multistart_sampling",
        )

    def forward(
        self,
        td: TensorDict,
        env: Union[str, RL4COEnvBase, None] = None,
        phase: str = "train",
        return_actions: bool = False,
        return_entropy: bool = False,
        return_init_embeds: bool = False,
        **decoder_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding
            phase: Phase of the algorithm (train, val, test)
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_init_embeds: Whether to return the initial embeddings
            decoder_kwargs: Keyword arguments for the decoder

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # ENCODER: get embeddings from initial state
        graph, init_embeds = self.encoder(td)

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase
        if decoder_kwargs.get("decode_type", None) is None:
            decoder_kwargs["decode_type"] = getattr(self, f"{phase}_decode_type")

        # DECODER: main rollout with autoregressive decoding
        log_p, actions, td_out = self.decoder(
            td, graph, env, phase=phase, **decoder_kwargs
        )

        out = {k: td_out[k] for k in ("reward", "heatmap")}

        if phase == "train":
            # Log likelihood is calculated within the model
            log_likelihood = get_log_likelihood(
                log_p, actions, td_out.get("mask", None)
            )  # , return_sum=False).mean(-1)
            out["log_likelihood"] = log_likelihood

        if return_actions:
            out["actions"] = actions

        if return_entropy:
            entropy = -(log_p.exp() * log_p).nansum(dim=1)  # [batch, decoder steps]
            entropy = entropy.sum(dim=1)  # [batch]
            out["entropy"] = entropy

        if return_init_embeds:
            out["init_embeds"] = init_embeds

        return out
