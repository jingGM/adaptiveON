from gym.spaces import Box, Dict
import numpy as np
import torch as th
from typing import Union


class ENVTYPE:
    STATIC = 0
    UNFLATTEN = 1
    HILLS = 2
    ALL = 3


class CONFIGTYPE:
    RAISIM = 0
    UNITY = 1


action_threshold = (1.0, 1.0)
action_linear_range = (0.0, action_threshold[0])
action_angular_range = (-action_threshold[1], action_threshold[1])
acceleration_max = (3., 5.)
lidar_range = (0., 10.)
lidar_shape = (341,)
elevation_map_shape = (80, 80)
imu_shape = (6,)

ActionSpace = Box(low=np.array([action_linear_range[0], action_angular_range[0]]),
                  high=np.array([action_linear_range[1], action_angular_range[1]]), shape=(2,), dtype=np.float32)
ActionThresholdSpace = Box(low=np.array([0., 0.]), high=np.array([action_threshold[0], action_threshold[1]]),
                           shape=(2,), dtype=np.float32)
ElevationSpace = Box(high=np.inf, low=-np.inf, shape=elevation_map_shape, dtype=np.float32)
LidarSpace = Box(high=lidar_range[1], low=lidar_range[0], shape=lidar_shape, dtype=np.float32)
StateSpace = Box(high=np.inf, low=-np.inf, shape=(11,), dtype=np.float32)
EnvironmentSpace = Box(high=np.inf, low=-np.inf, shape=(6,), dtype=np.float32)
EnvironmentActionSpace = Box(low=np.array([[action_linear_range[0], action_angular_range[0]],
                                           [action_linear_range[0], action_angular_range[0]],
                                           [action_linear_range[0], action_angular_range[0]]]),
                             high=np.array([[action_linear_range[1], action_angular_range[1]],
                                            [action_linear_range[1], action_angular_range[1]],
                                            [action_linear_range[1], action_angular_range[1]]]),
                             shape=(3, 2), dtype=np.float32)
EnvironmentStateSpace = Box(high=np.inf, low=-np.inf, shape=(3, 128), dtype=np.float32)
StateOutputSpace = Box(high=np.inf, low=-np.inf, shape=(128,), dtype=np.float32)
EnvironmentOutputSpace = Box(high=np.inf, low=-np.inf, shape=(64,), dtype=np.float32)
GoalSpace = Box(high=np.inf, low=-np.inf, shape=(3,), dtype=np.float32)
PoseSpace = Box(high=np.inf, low=-np.inf, shape=(6,), dtype=np.float32)

ObservationSpace = Dict(
    {
        "map_c": ElevationSpace,
        "map_1": ElevationSpace,
        "map_2": ElevationSpace,
        "map_3": ElevationSpace,

        "lidar_c": LidarSpace,
        # "lidar_1": LidarSpace,
        # "lidar_2": LidarSpace,
        # "lidar_3": LidarSpace,

        "state_c": StateSpace,  # pose, goal, vel
        "state_1": StateSpace,  # pose, goal, vel
        "state_2": StateSpace,  # pose, goal, vel
        "state_3": StateSpace,  # pose, goal, vel

        "environments": EnvironmentSpace,

        # "action_c": ActionSpace,
        "action_1": ActionSpace,
        "action_2": ActionSpace,
        "action_3": ActionSpace,

        "goal": GoalSpace,
        "pose": PoseSpace
    }
)

SingleObservationSpace = Dict(
    {
        "elevation": ElevationSpace,
        "lidar": LidarSpace,
        "state": StateSpace,  # pose, goal, vel
        "environments": EnvironmentSpace,
        "last_action": ActionSpace,
    }
)


class SingleObservation:
    def __init__(self,
                 lidar: Union[np.array, th.tensor] = None,
                 map: Union[np.array, th.tensor] = None,
                 environments: Union[np.array, th.tensor] = None,
                 last_action: Union[np.array, th.tensor] = None,
                 state: Union[np.array, th.tensor] = None,
                 lidar_poses: Union[np.array, th.tensor] = None,
                 world_pose: Union[np.array, th.tensor] = None,
                 euclidean_goal: Union[np.array, th.tensor] = None):
        self.lidar = lidar
        self.map = map
        self.state = state
        self.environments = environments
        self.last_action = last_action

        self.euclidean_goal = euclidean_goal
        self.world_pose = world_pose
        self.lidar_poses = lidar_poses

    def get_observation(self):
        return {"elevation": self.map,
                "lidar": self.lidar,
                "state": self.state,
                "environments": self.environments,
                "last_action": self.last_action,
                }
