from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import torch as th
from gym import spaces
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class Distribution:
    def __init__(self, action_dim: int,
                 full_std: bool = True,
                 use_expln: bool = False,
                 # squash_output: bool = False,
                 learn_features: bool = False,
                 epsilon: float = 1e-6,
                 use_sde: bool = False
                 ):
        self.distribution = None
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None
        self.std = None
        self.use_sde = use_sde

        self.latent_sde_dim = None
        self.weights_dist = None
        self.exploration_mat = None
        self.exploration_matrices = None
        self._latent_sde = None
        self.use_expln = use_expln
        self.full_std = full_std
        self.epsilon = epsilon
        self.learn_features = learn_features
        # if squash_output:
        #     self.bijector = TanhBijector(epsilon)
        # else:
        #     self.bijector = None

    def sample_weights(self, log_std: th.Tensor, batch_size: int = 1) -> None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.
        """
        std = self.get_std(log_std)
        self.weights_dist = Normal(th.zeros_like(std), std)
        # Reparametrization trick to pass gradients
        self.exploration_mat = self.weights_dist.rsample()
        # Pre-compute matrices in case of parallel exploration
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def get_std(self, log_std: th.Tensor) -> th.Tensor:
        if self.use_expln:
            below_threshold = th.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (th.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = th.exp(log_std)

        if self.full_std:
            return std
        return th.ones(self.latent_sde_dim, self.action_dim).to(log_std.device) * std

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()

    def proba_distribution_net(self, latent_dim: int,
                               log_std_init: float = None,
                               latent_sde_dim: Optional[int] = None) -> Tuple[nn.Module, nn.Parameter]:
        mean_actions = nn.Linear(latent_dim, self.action_dim)

        if log_std_init is None:
            if self.use_sde:
                log_std_init = -2.0
            else:
                log_std_init = 0.0

        if self.use_sde:
            self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
            # Reduce the number of parameters if needed
            log_std = th.ones(self.latent_sde_dim, self.action_dim) if self.full_std else th.ones(self.latent_sde_dim,
                                                                                                  1)
            # Transform it to a parameter so it can be optimized
            log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
            # Sample an exploration matrix
            self.sample_weights(log_std)
        else:
            log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        self.log_std = log_std
        return mean_actions, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor) -> "Distribution":
        if self.use_sde:
            self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
            action_std = self.get_std(log_std)
            variance = th.mm(self._latent_sde ** 2, action_std ** 2)
            self.distribution = Normal(mean_actions, th.sqrt(variance + self.epsilon))
        else:
            action_std = th.ones_like(mean_actions) * log_std.exp()
            self.distribution = Normal(mean_actions, action_std)

        self.log_std = log_std
        self.std = action_std
        self.mean_actions = mean_actions
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        if self.use_sde:
            noise = self.get_noise(self._latent_sde)
            return self.distribution.mean + noise
        else:
            return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def get_noise(self, latent_sde: th.Tensor) -> th.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return th.mm(latent_sde, self.exploration_mat)
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(1)
        # (batch_size, 1, n_actions)
        noise = th.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(1)

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor,
                            deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(mean_actions, log_std, latent_sde)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor,
                             latent_sde: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std, latent_sde)
        log_prob = self.log_prob(actions)
        return actions, log_prob
