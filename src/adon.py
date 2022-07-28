import torch as th
from torch import nn
from functools import partial
from typing import Union, Tuple, List, Dict
import numpy as np
import gym
from src.distribution import Distribution
from src.utils import get_device, obs_as_tensor
from src.networks import ActCritic, ModelType
from src.configs import ActionThresholdSpace


class ACPolicy(nn.Module):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, model: ActCritic):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.threshold_space = ActionThresholdSpace
        self.mlp_extractor = model

        self.action_dist = Distribution(action_dim=int(np.prod(action_space.shape)))
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi)

        self.stable_dist = Distribution(action_dim=int(np.prod(action_space.shape)))
        latent_dim_stable = self.mlp_extractor.latent_dim_stable
        self.threshold_net, self.log_stable_std = self.stable_dist.proba_distribution_net(
            latent_dim=latent_dim_stable, latent_sde_dim=latent_dim_stable)

    def reset(self):
        pass

    @classmethod
    def load_model(cls, path: str, model_type: List[int], observation_space: gym.spaces.Space, model: ActCritic,
                   action_space: gym.spaces.Space, device: Union[th.device, str] = "cuda") -> "ACPolicy":
        device = get_device(device)
        saved_variables = th.load(path, map_location=device)

        model = cls(observation_space=observation_space, action_space=action_space, model=model)

        if ModelType.POLICY in model_type:
            model.mlp_extractor.value_net.load_state_dict(saved_variables["state_dict"]["value_net"])
            model.mlp_extractor.policy_net.load_state_dict(saved_variables["state_dict"]["policy_net"])
            model.action_net.load_state_dict(saved_variables["state_dict"]["action_net"])

        if ModelType.STABLE in model_type:
            model.mlp_extractor.stable_net.load_state_dict(saved_variables["state_dict"]["stable_net"])
            model.mlp_extractor.stable_value_net.load_state_dict(saved_variables["state_dict"]["stable_value_net"])
            model.threshold_net.load_state_dict(saved_variables["state_dict"]["threshold_net"])

        if ModelType.STATIC in model_type:
            model.mlp_extractor.lidar_model.load_state_dict(saved_variables["state_dict"]["lidar_model"])

        if ModelType.TERRAIN in model_type:
            model.mlp_extractor.elevation_model.load_state_dict(saved_variables["state_dict"]["elevation_model"])

        if ModelType.ENVIRONMENT in model_type:
            model.mlp_extractor.adaptive_model.load_state_dict(saved_variables["state_dict"]["adaptive_model"])

        model.to(device)
        return model

    def save(self, path: str) -> None:
        states = {"action_net": self.action_net.state_dict(),
                  "log_std": self.state_dict()['log_std'],
                  "value_net": self.mlp_extractor.value_net.state_dict(),
                  "policy_net": self.mlp_extractor.policy_net.state_dict(),

                  "threshold_net": self.threshold_net.state_dict(),
                  "log_stable_std": self.state_dict()['log_stable_std'],
                  "stable_value_net": self.mlp_extractor.stable_value_net.state_dict(),
                  "stable_net": self.mlp_extractor.stable_net.state_dict(),

                  "lidar_model": self.mlp_extractor.lidar_model.state_dict(),
                  "elevation_model": self.mlp_extractor.elevation_model.state_dict(),
                  }
        states.update({"adaptive_model": self.mlp_extractor.adaptive_model.state_dict()})
        th.save({"state_dict": states, "data": self._get_constructor_parameters()}, path)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)

    def _get_stable_dist_from_latent(self, latent_stable: th.Tensor) -> Distribution:
        mean_stable = self.threshold_net(latent_stable)
        return self.stable_dist.proba_distribution(mean_stable, self.log_stable_std, latent_stable)

    def _handle_collision(self, lidar):
        lidar = lidar[0] * 10
        threshold = 0.7
        min_dis = np.min(lidar)
        indeces = np.where(lidar < threshold)[0]
        left = len(np.intersect1d(np.where(indeces < 341 * 4 / 5)[0], np.where(indeces > 341 / 2)[0]))
        right = len(np.intersect1d(np.where(indeces > 341 / 5)[0], np.where(indeces < 341 / 2)[0]))
        # if min_dis < threshold:
        #     print("test")
        if min_dis < threshold and (left > 0 or right > 0):
            if left > right:
                return [0., -1]
            else:
                return [0., 1]
        return None

    def predict(self, observation: Union[np.ndarray, Dict[str, np.ndarray]], deterministic: bool = True) -> \
            Tuple[th.Tensor, th.Tensor]:
        self.train(False)
        action = self._handle_collision(observation["lidar_c"])
        observation_tensor = obs_as_tensor(observation, self.mlp_extractor.device)
        with th.no_grad():
            latent_pi, latent_stable = self.mlp_extractor.forward_actor(observation_tensor)
            actions = self._get_action_dist_from_latent(latent_pi=latent_pi).get_actions(deterministic)
            thresholds = self._get_stable_dist_from_latent(latent_stable=latent_stable).get_actions(deterministic)
        actions = actions.cpu().numpy()
        thresholds = thresholds.cpu().numpy()
        thresholds = self.get_action(observation=observation, actions=thresholds)

        if isinstance(self.action_space, gym.spaces.Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        if isinstance(self.threshold_space, gym.spaces.Box):
            thresholds = np.clip(thresholds, self.threshold_space.low, self.threshold_space.high)
        if action is not None:
            actions = np.array([action])
        return actions, thresholds

    def get_action(self, observation, actions):
        stable_reward_threshold = 0.3
        last_imu_0 = np.abs(observation["state_1"][0][2:4])
        last_imu_1 = np.abs(observation["state_2"][0][2:4])
        last_imu_2 = np.abs(observation["state_3"][0][2:4])
        last_imu_c = np.abs(observation["state_c"][0][2:4])
        total_rp = np.sum(last_imu_0 + last_imu_1 + last_imu_2 + last_imu_c) / 4.
        print("vib: {:.2f}".format(max(total_rp, 0)), end="    ")
        if total_rp > 0.3:
            total_rp += 0.2
        else:
            total_rp = total_rp
        sum_rp = min(max(total_rp, 0), 1.0)
        if sum_rp < stable_reward_threshold:
            actions[0][0] = 1.0
            actions[0][1] = 1.0
        else:
            v_threshold = stable_reward_threshold / sum_rp
            actions[0][0] = v_threshold
            actions[0][1] = v_threshold
        return actions
