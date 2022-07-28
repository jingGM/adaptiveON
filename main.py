import os
import numpy as np
import rospy
import math
import argparse

from src.env import Environment
import torch as th
from src.utils import get_device
from src.networks import ActCritic, ModelType
from src.adon import ACPolicy


def get_args():
    parser = argparse.ArgumentParser(description='Adaptive RL')
    parser.add_argument('--exp_name', type=str, default="empty", help="the name of the experiment")

    parser.add_argument('--en_odom_position', action='store_true', default=False,
                        help="Use the odometery's position or SLAM")
    parser.add_argument('--en_lidar', action='store_false', default=True, help="Use Lidar")
    parser.add_argument('--en_elevation', action='store_false', default=True, help="Use Elevation map package")
    parser.add_argument('--imu_orientation', action='store_false', default=True,
                        help="Use IMU for orientation detection")

    parser.add_argument('--device', type=str, default='cuda', help="cuda or cpu")
    parser.add_argument('--goal', nargs='+', help='<Required> Set flag', required=True, type=float)
    parser.add_argument('--successful_threshold', type=float, default=1, help="threshold to stop around the goal, (m)")
    parser.add_argument('--collision_threshold', type=float, default=0.4,
                        help="minimum distance threshold to stop around obstacles, meter")
    parser.add_argument('--time_step', type=float, default=0.1, help="estimated time step, (s)")
    parser.add_argument('--root_dir', type=str, default="./", help="root directory of the project")
    parser.add_argument('--load_model_dir', type=str, default="./", help="the directory to the trained file")
    return parser.parse_args()


def get_adptRL(argument):
    device = get_device(argument.device)
    model = ActCritic(device=device, train=argument.train, network_mode=argument.model_type)
    policy = ACPolicy.load_model(argument.load_model_dir, model_type=argument.load_model_type, model=model,
                                 observation_space=argument.observation_space, action_space=argument.action_space)
    policy.train(False)
    return policy


def convert_obs(obs):
    observation = {}
    # print(obs)
    for (key, value) in obs.items():
        # print(key)
        observation[key] = np.expand_dims(value, axis=0)
    return observation


def run(policy, environment):
    obs = environment.reset()
    done = False
    while not done:
        obs = convert_obs(obs)
        actions, thresholds = policy.predict(observation=obs, deterministic=True)
        actions = actions[0]
        thresholds = thresholds[0]
        obs, done, info = environment.step(np.array(actions), thresholds)


if __name__ == "__main__":
    rospy.init_node("adpt_sim")
    args = get_args()
    th.autograd.set_detect_anomaly(True)
    args.goal = [args.goal[0], args.goal[1], 0]

    args.root_dir = "/home/gamma-robot/adpt_ws/src/adpt_sim"

    if not os.path.exists(args.load_model_dir):
        Exception("the model path doesn't exist")

    args.model_type = [ModelType.POLICY, ModelType.STABLE, ModelType.ENVIRONMENT, ModelType.STATIC, ModelType.TERRAIN]
    args.load_model_type = [ModelType.POLICY, ModelType.ENVIRONMENT, ModelType.STATIC, ModelType.TERRAIN,
                            ModelType.STABLE]

    env = Environment(en_odom_position=args.en_odom_position, goal=args.goal, terminate_stop=True,
                      enable_render=True, time_step=args.time_step, successful_threshold=args.successful_threshold,
                      collision_threshold=args.collision_threshold, stable_threshold=40 * math.pi / 180.,
                      en_lidar=args.en_lidar, en_elevation=args.en_elevation, imu_orientation=args.imu_orientation,
                      experiment_name=args.exp_name, directory=args.root_dir)
    args.observation_space = env.observation_space
    args.action_space = env.action_space

    expert_ply = get_adptRL(args)
    run(policy=expert_ply, environment=env)
