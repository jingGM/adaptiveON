import torch as th
from torch import nn
from functools import partial
from typing import Union, Tuple
from typing import Dict as Dict_type
from src.configs import ElevationSpace, LidarSpace, StateSpace, EnvironmentSpace, SingleObservation, \
    EnvironmentActionSpace, ActionSpace
import numpy as np

LayerTypes = [nn.Conv2d, nn.Conv1d, nn.Linear, nn.Conv3d, nn.LSTM, nn.GRU]


class ModelType:
    POLICY = 0
    STATIC = 1  # only lidar
    TERRAIN = 2  # only elevation map
    ENVIRONMENT = 3  # only environment
    STABLE = 4


class ActCritic(nn.Module):
    def __init__(self, device, network_mode: [int], train: bool = True):
        super(ActCritic, self).__init__()
        self.latent_dim_pi = 64
        self.latent_dim_stable = 64
        self.en_train = train
        self.device = device
        self.env_param_mode = False
        self.mode = network_mode
        if not self.en_train:
            self.history_actions = None
            self.history_states = None
            self.history_observations = None

        act_function = nn.ReLU

        elevation_output = 64
        self.elevation_model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=(2, 2)), act_function(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(5, 5), stride=(2, 2)), act_function(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), stride=(1, 1)), act_function(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1)), act_function(),
            nn.Flatten(), nn.Linear(in_features=2304, out_features=512), act_function(),
            nn.Linear(in_features=512, out_features=256), act_function(),
            nn.Linear(in_features=256, out_features=elevation_output), act_function()
        )

        lidar_output = 128
        self.lidar_model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(5,), stride=(1,)), act_function(),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(3,), stride=(1,)), act_function(),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(3,), stride=(1,)), act_function(),
            nn.Flatten(), nn.Linear(in_features=333, out_features=256), act_function(),
            nn.Linear(in_features=256, out_features=lidar_output), act_function(),
        )
        current_state_output = lidar_output + elevation_output + StateSpace.shape[0]

        environment_output = 64
        if self.env_param_mode:
            self.environment_model = nn.Sequential(
                nn.Linear(EnvironmentSpace.shape[0] + StateSpace.shape[0] * 3, 128), act_function(),
                nn.Linear(128, 64), act_function(),
                nn.Linear(64, environment_output), act_function(),
            )
        else:
            self.adaptive_model = nn.Sequential(
                nn.Linear(ActionSpace.shape[0] * 3 + StateSpace.shape[0] * 3, 128), act_function(),
                nn.Linear(128, 64), act_function(),
                nn.Linear(64, environment_output), act_function(),
            )

        self.stable_net = nn.Sequential(nn.Linear(ActionSpace.shape[0] * 3 + StateSpace.shape[0] * 3, 128),
                                        act_function(),
                                        nn.Linear(128, 64), act_function(),
                                        nn.Linear(64, self.latent_dim_stable), act_function())
        self.stable_value_net = nn.Sequential(nn.Linear(ActionSpace.shape[0] * 3 + StateSpace.shape[0] * 3, 128),
                                              act_function(),
                                              nn.Linear(128, 64), act_function(), nn.Linear(64, 1))

        self.policy_net = nn.Sequential(nn.Linear(environment_output + current_state_output, 128),
                                        act_function(), nn.Linear(128, 64), act_function(),
                                        nn.Linear(64, self.latent_dim_pi), act_function())
        self.value_net = nn.Sequential(nn.Linear(environment_output + current_state_output, 128),
                                       act_function(), nn.Linear(128, 64), act_function(), nn.Linear(64, 1))

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def initialize(self):
        module_gains = {}
        if ModelType.POLICY in self.mode:
            module_gains.update({self.policy_net: np.sqrt(2), self.value_net: np.sqrt(2)})
        else:
            for param in self.policy_net.parameters():
                param.requires_grad = False
            for param in self.value_net.parameters():
                param.requires_grad = False
            module_gains.update({self.policy_net: 0, self.value_net: 0})

        if ModelType.STABLE in self.mode:
            module_gains.update({self.stable_net: np.sqrt(2), self.stable_value_net: np.sqrt(2)})
        else:
            for param in self.stable_net.parameters():
                param.requires_grad = False
            for param in self.stable_value_net.parameters():
                param.requires_grad = False
            module_gains.update({self.stable_net: 0, self.stable_value_net: 0})

        if ModelType.STATIC in self.mode:
            module_gains.update({self.lidar_model: np.sqrt(2)})
        else:
            for param in self.lidar_model.parameters():
                param.requires_grad = False
            module_gains.update({self.lidar_model: 0})

        if ModelType.TERRAIN in self.mode:
            module_gains.update({self.elevation_model: np.sqrt(2)})
        else:
            for param in self.elevation_model.parameters():
                param.requires_grad = False
            module_gains.update({self.elevation_model: 0})

        if ModelType.ENVIRONMENT in self.mode:
            environment_value = np.sqrt(2)
            if self.env_param_mode:
                module_gains.update({self.environment_model: environment_value, })
            else:
                module_gains.update({self.adaptive_model: environment_value, })
        else:
            environment_value = np.sqrt(0)
            # for param in self.dynamic_preprocess.parameters():
            #     param.requires_grad = False
            if self.env_param_mode:
                module_gains.update({self.environment_model: environment_value, })
                for param in self.environment_model.parameters():
                    param.requires_grad = False
            else:
                module_gains.update({self.adaptive_model: environment_value, })
                for param in self.adaptive_model.parameters():
                    param.requires_grad = False

        for module, gain in module_gains.items():
            module.apply(partial(self.init_weights, gain=gain))

    def process_states(self, observations):
        assert StateSpace.shape == observations["state"].shape[1:], "The state shape is not correct"
        assert LidarSpace.shape == observations["lidar"].shape[1:], "The lidar shape is not correct"
        assert ElevationSpace.shape == observations["elevation"].shape[1:], "The elevation shape is not correct"
        observations["lidar"] = observations["lidar"][:, None, :]
        observations["elevation"] = observations["elevation"][:, None, :, :]
        lidar = self.lidar_model.forward(observations["lidar"])
        elevation = self.elevation_model(observations["elevation"])
        state_input = th.concat((lidar, elevation), dim=1)
        return state_input

    def process_observations(self, obs: Union[Dict_type[str, th.Tensor], th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        state_observation_c = SingleObservation(lidar=obs["lidar_c"], map=obs["map_c"], state=obs["state_c"])
        current_obs = self.process_states(state_observation_c.get_observation())
        current_state = obs["state_c"]
        # state_observation_1 = SingleObservation(lidar=obs["lidar_1"], map=obs["map_1"])
        # state_observation_2 = SingleObservation(lidar=obs["lidar_2"], map=obs["map_2"])
        # state_observation_3 = SingleObservation(lidar=obs["lidar_3"], map=obs["map_3"])
        # obs_1 = self.process_states(state_observation_1.get_observation())
        # obs_2 = self.process_states(state_observation_2.get_observation())
        # obs_3 = self.process_states(state_observation_3.get_observation())
        # previous_observations = th.concat((obs_1[:, None, :], obs_2[:, None, :], obs_3[:, None, :]), dim=1)
        # observations = self.dynamic_preprocess(previous_observations)
        previous_states = th.concat((obs["state_1"], obs["state_2"], obs["state_3"]), dim=1)

        previous_actions = th.concat((obs["action_1"], obs["action_2"], obs["action_3"]), dim=1)
        assert previous_actions.shape[1:] == (EnvironmentActionSpace.shape[0] * EnvironmentActionSpace.shape[
            1],), "The previous actions shape isn't correct"
        stable_input = th.concat((previous_states, previous_actions), dim=-1)
        current_environment = self.adaptive_model(stable_input)
        return th.concat((current_obs, current_state, current_environment), -1), stable_input

    def forward(self, obs: Union[Dict_type[str, th.Tensor], th.Tensor]) -> Tuple[th.Tensor, th.Tensor, th.Tensor,
                                                                                 th.Tensor]:
        obs, stable_input = self.process_observations(obs)
        return self.policy_net(obs), self.value_net(obs), self.stable_net(stable_input), self.stable_value_net(stable_input)

    def forward_actor(self, obs: Union[Dict_type[str, th.Tensor], th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        obs, stable_input = self.process_observations(obs)
        return self.policy_net(obs), self.stable_net(stable_input)

    def forward_critic(self, obs: Union[Dict_type[str, th.Tensor], th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        obs, stable_input = self.process_observations(obs)
        return self.value_net(obs), self.stable_value_net(stable_input)
