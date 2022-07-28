import rospy
import copy
import math
import time
from typing import List, Tuple, Union
import warnings
import numpy as np
from scipy.interpolate import griddata

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import String

from src.configs import SingleObservation, ObservationSpace, ActionSpace, ActionThresholdSpace, elevation_map_shape, \
    lidar_shape, lidar_range, action_threshold, imu_shape


def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class TOPICS:
    odom_topic = '/odometry/filtered'
    imu_topic = "/zed2i/zed_node/imu/data"
    scan_topic = '/scan'
    elevation_map_topic = '/grid_map_visualization/elevation_grid'
    reset_topic = '/unity_command/command_topic'
    action_publisher_topic = "/cmd_vel"  # "/jackal_velocity_controller/cmd_vel" '/cmd_vel'
    localization_topic = "/integrated_to_init"


def _get_goal_location(pose: np.ndarray, goal: np.ndarray):
    """
    @ pose: [x, y, z, yaw]
    @ goal: [x, y, z]
    """
    assert pose.shape == (4,), "the pose shape is not correct {}".format(pose.shape)
    assert goal.shape == (3,), "the goal shape is not correct {}".format(goal.shape)
    alpha = pose[3]
    position = np.array(goal[:2]) - np.array(pose[0:2])
    rotation = np.array([[math.cos(alpha), math.sin(alpha)],
                         [-math.sin(alpha), math.cos(alpha)]])
    rel_pose = np.matmul(rotation, position)
    return [np.linalg.norm(rel_pose), math.atan2(rel_pose[1], rel_pose[0]), goal[2] - pose[2]]


class Environment:
    def __init__(self, goal, terminate_stop=True, enable_render=True, time_step=0.1, en_odom_position=False,
                 successful_threshold=1.0, collision_threshold=0.5, stable_threshold=40. * math.pi / 180.,
                 en_lidar=True, en_elevation=True, imu_orientation=False, experiment_name=""):
        self.action_space = ActionSpace
        self.observation_space = ObservationSpace
        self.action_threshold_space = ActionThresholdSpace

        self.n_step = 0
        self.time_step = time_step
        self.elevation_map_height_threshold = 2
        self.enable_render = enable_render
        self.terminate_stop = terminate_stop
        self.successful_threshold = successful_threshold
        self.collision_threshold = collision_threshold
        self.stable_threshold = stable_threshold
        self.starting_time = time.time()

        self.lidar_min_dis = math.inf
        self.action_threshold = [action_threshold[0], action_threshold[1]]
        self.en_odom_position = en_odom_position
        self.en_lidar = en_lidar
        self.en_elevation = en_elevation
        self.imu_orientation = imu_orientation
        self.current_pose = None
        self.current_imu = None
        self.current_lidar = None
        self.current_elevation_map = None
        self.current_vel = None
        self.robot_terminate = False
        self.first_pose = None
        self.goal = np.array(goal)
        self.relative_goal = None
        self.last_obs: List[SingleObservation] = []
        self.last_imus = [np.zeros(imu_shape), np.zeros(imu_shape), np.zeros(imu_shape), np.zeros(imu_shape)]

        self.current_imu_raw = None
        self.current_elevation_map_raw = None
        self.current_lidar_raw = None
        self.current_position_raw = None
        self.current_orientation_raw = None
        self.current_vel_raw = None
        self.odom_pose = None

        self.experiment_name = experiment_name

        self._build_ros_interfaces()

    def reset(self):
        self.last_imus = [np.zeros(imu_shape), np.zeros(imu_shape), np.zeros(imu_shape), np.zeros(imu_shape)]
        while not self._topic_status():
            print("waiting for the topics 0.1s")
            time.sleep(0.1)

        observation = self.get_observation(current_action=np.array([0., 0.]))
        return observation

    def step(self, action: Union[list, np.ndarray], vel_threshold=None):
        if self.terminate_stop and self.robot_terminate:
            action = np.zeros(len(action))
        action = self._adjust_action(action=action)

        if vel_threshold is None:
            self.action_threshold = [self.action_space.high[0], self.action_space.high[1]]
        else:
            temp_action_threshold = self._adjust_action_threshold(action=vel_threshold)
            self.action_threshold[0] = temp_action_threshold[0]
            self.action_threshold[1] = temp_action_threshold[1]
            action[0] = action[0] * self.action_threshold[0] / self.action_space.high[0]
            action[1] = action[1] * self.action_threshold[1] / self.action_space.high[1]

        assert action.shape == ActionSpace.shape, "action type is not correct"
        twist_action = Twist()
        twist_action.linear.x = action[0]
        twist_action.angular.z = action[1]
        self.pub_cmd_vel.publish(twist_action)
        if time.time() - self.starting_time < self.time_step:
            pass

        self.n_step += 1
        observation = self.get_observation(current_action=np.zeros(self.action_space.shape))
        self._check_status()

        current_velocity = np.array([self.current_vel[0], self.current_vel[5]])

        self.last_imus.pop(0)
        self.last_imus.append(copy.deepcopy(self.current_imu))

        if self.enable_render:
            print("t:{}".format(time.time() - self.starting_time), end=", ")
            print("s:{}".format(self.n_step), end=", ")
            print("c:{}".format(self.lidar_min_dis), end=", ")
            print("a:[{:.2f}, {:.2f}]".format(action[0], action[1]), end=", ")
            print("th:[{:.2f}, {:.2f}]".format(self.action_threshold[0], self.action_threshold[1]), end=", ")
            print("g:[{:.2f},{:.2f}]".format(self.relative_goal[0], self.relative_goal[1]), end=", ")
            print("ag:[{:.2f},{:.2f}]".format(self.goal[0], self.goal[1]), end=", ")
            print("v:[{:.2f}, {:.2f}]".format(current_velocity[0], current_velocity[1]), end=", ")
            print("imu:[", end="")
            for imu in self.current_imu:
                print("{:.2f}".format(imu), end=",")
            print("]", end=", ")
            print("p:[", end="")
            for pos in self.current_pose:
                print("{:.2f}".format(pos), end=",")
            print("]")
        self.starting_time = time.time()
        return observation, self.robot_terminate, {}

    def get_observation(self, current_action=None):
        if current_action is None:
            current_action = np.zeros(self.action_space.shape)

        self._get_robot_state()
        self._test_state()

        self.current_lidar, self.current_elevation_map = self._get_lidar_map_observation()
        # np.save("elevation_map", np.array(self.current_elevation_map))
        self.current_lidar = self.current_lidar  # / lidar_range[1]
        self._test_observation()

        self.relative_goal = _get_goal_location(
            pose=np.array([self.current_pose[0], self.current_pose[1], self.current_pose[2], self.current_pose[5]]),
            goal=self.goal)
        linear_vel = np.linalg.norm(self.current_vel[0:3])
        angular_vel = self.current_vel[5]

        sobs = SingleObservation(lidar=self.current_lidar, map=self.current_elevation_map,
                                 last_action=current_action,
                                 state=np.concatenate((self.current_imu, self.relative_goal,
                                                       np.array([linear_vel, angular_vel]))))
        return self._generate_training_obs(sobs)

    def _build_ros_interfaces(self):
        self.pub_relocate_robot = rospy.Publisher(TOPICS.reset_topic, String, queue_size=10)
        self.pub_cmd_vel = rospy.Publisher(TOPICS.action_publisher_topic, Twist, queue_size=10)

        self.sub_odom = rospy.Subscriber(TOPICS.odom_topic, Odometry, self._odom_callback)
        if not self.en_odom_position:
            self.sub_localization = rospy.Subscriber(TOPICS.localization_topic, Odometry, self._position_callback)
        self.sub_imu = rospy.Subscriber(TOPICS.imu_topic, Imu, self._imu_callback)
        if self.en_lidar:
            self.sub_scan = rospy.Subscriber(TOPICS.scan_topic, LaserScan, self._scan_callback)
        if self.en_elevation:
            self.sub_elevation_map = rospy.Subscriber(TOPICS.elevation_map_topic, OccupancyGrid,
                                                      self._elevation_map_callback)

    def _relative_pose_to_first_frame(self, pose):
        # pose: x, y, z, roll, pitch, yaw
        relative_xy = np.array(pose[:2]) - np.array(self.first_pose[:2])
        alpha = self.first_pose[5]
        rotation = np.array([[math.cos(alpha), math.sin(alpha)], [-math.sin(alpha), math.cos(alpha)]])
        rel_pose = np.matmul(rotation, relative_xy)
        return np.array([rel_pose[0], rel_pose[1], pose[2] - self.first_pose[2], pose[3], pose[4], pose[5] - alpha])

    def _topic_status(self):
        tag = True
        print("topics empty: ", end=": ")
        if self.current_vel_raw is None:
            print("vel", end=", ")
            tag = False
        if self.en_lidar:
            if self.current_lidar_raw is None:
                print("lidar", end=", ")
                tag = False
        if self.en_elevation:
            if self.current_elevation_map_raw is None:
                print("elevation", end=", ")
                tag = False
        if self.current_position_raw is None:
            print("position", end=", ")
            tag = False
        if self.current_orientation_raw is None:
            print("orientation", end=", ")
            tag = False
        if self.current_imu_raw is None:
            print("imu", end=", ")
            tag = False
        print(" ")
        return tag

    def _elevation_map_callback(self, raw_data: OccupancyGrid):
        map_data = np.array(raw_data.data)
        map_width = raw_data.info.width
        map_height = raw_data.info.height
        map2d = np.flip(np.rot90(np.reshape(map_data, (map_height, map_width))), axis=0)
        result2 = np.where(map2d != -1)

        values = map2d[result2]
        points = np.vstack((result2[0], result2[1])).T
        grid_x, grid_y = np.meshgrid(np.arange(0, map_width, 1), np.arange(0, map_height, 1))

        interpolated_grid = griddata(points, values, (grid_x, grid_y), 'nearest')
        self.current_elevation_map_raw = interpolated_grid.astype('float64')

    def _get_elevation_map(self):
        if self.current_elevation_map_raw is None:
            elevation_map = np.ones(elevation_map_shape)
        else:
            elevation_map = np.array(self.current_elevation_map_raw) / self.elevation_map_height_threshold
            elevation_map = np.where(elevation_map <= 1.0, elevation_map, 1.0)
            elevation_map = np.where(elevation_map == np.inf, 0, elevation_map)
            elevation_map = np.where(elevation_map >= -1.0, elevation_map, -1.0)
        return elevation_map

    def _scan_callback(self, raw_data: LaserScan):
        self.current_lidar_raw = raw_data.ranges

    def _get_lidar(self):
        if self.current_lidar_raw is None:
            lidar = np.ones(lidar_shape) * lidar_range[1]
        else:
            unity_range = 723
            index = int((unity_range - lidar_shape[0]) / 2)
            lidar = np.array(self.current_lidar_raw[index: unity_range - index])
            lidar = np.where(lidar > lidar_range[1], lidar_range[1], lidar)
        self.lidar_min_dis = np.min(lidar)
        return lidar / lidar_range[1]

    def _imu_callback(self, raw_data: Imu):
        linear_acc = np.linalg.norm([raw_data.linear_acceleration.x, raw_data.linear_acceleration.y])
        roll, pitch, yaw = euler_from_quaternion(raw_data.orientation.x, raw_data.orientation.y,
                                                 raw_data.orientation.z, raw_data.orientation.w)
        self.current_imu_raw = [linear_acc, raw_data.linear_acceleration.z,
                                raw_data.angular_velocity.x, raw_data.angular_velocity.y, roll, pitch]
        if self.imu_orientation:
            self.current_orientation_raw = [roll, pitch, yaw]

    def _odom_callback(self, raw_data: Odometry):
        position = raw_data.pose.pose.position
        orientation = raw_data.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w)
        self.current_vel_raw = [raw_data.twist.twist.linear.x, raw_data.twist.twist.linear.y,
                                raw_data.twist.twist.linear.z,
                                raw_data.twist.twist.angular.x, raw_data.twist.twist.angular.y,
                                raw_data.twist.twist.angular.z]
        self.odom_pose = [position.x, position.y, position.z, roll, pitch, yaw]
        if self.en_odom_position:
            self.current_position_raw = [position.x, position.y, position.z]
        if not self.imu_orientation:
            self.current_orientation_raw = [roll, pitch, yaw]

    def _position_callback(self, raw_data: Odometry):
        position = raw_data.pose.pose.position
        # orientation = raw_data.pose.pose.orientation
        # roll, pitch, yaw = euler_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w)
        self.current_position_raw = [position.x, position.y, position.z]

    def _generate_training_obs(self, current_obs: SingleObservation):
        if len(self.last_obs) == 0:
            self.last_obs = [copy.deepcopy(current_obs), copy.deepcopy(current_obs), copy.deepcopy(current_obs)]
        observe = {
            "map_c": current_obs.map,
            "lidar_c": current_obs.lidar,
            # "lidar_1": self.last_obs[0].lidar,
            # "lidar_2": self.last_obs[1].lidar,
            # "lidar_3": self.last_obs[2].lidar,

            "state_c": current_obs.state,
            "state_1": self.last_obs[0].state,
            "state_2": self.last_obs[1].state,
            "state_3": self.last_obs[2].state,

            "action_1": self.last_obs[1].last_action,
            "action_2": self.last_obs[2].last_action,
            "action_3": current_obs.last_action,

            "goal": self.goal,
            "pose": self.current_pose
        }
        self.last_obs.pop(0)
        self.last_obs.append(current_obs)
        return observe

    def _adjust_action(self, action: list):
        assert np.array(action).shape == self.action_space.shape, "the action shape is wrong: {}".format(len(action))
        if np.isnan(action[0]):
            action[0] = 0
            warnings.warn("linear action is NAN")
        if np.isnan(action[1]):
            action[1] = 0
            warnings.warn("angular action is NAN")
        if action[0] < self.action_space.low[0] or action[0] > self.action_space.high[0] or \
                action[1] < self.action_space.low[1] or action[1] > self.action_space.high[1]:
            warnings.warn("angular action is out of range: {}".format(action))
            action = np.clip(a=action, a_min=self.action_space.low, a_max=self.action_space.high)
        return action

    def _adjust_action_threshold(self, action: list):
        assert np.array(action).shape == self.action_space.shape, "the threshold shape is wrong: {}".format(len(action))
        if np.isnan(action[0]):
            action[0] = 0
            warnings.warn("linear threshold is NAN")
        if np.isnan(action[1]):
            action[1] = 0
            warnings.warn("angular threshold is NAN")
        if action[0] < self.action_threshold_space.low[0] or action[0] > self.action_threshold_space.high[0] or \
                action[1] < self.action_threshold_space.low[1] or action[1] > self.action_threshold_space.high[1]:
            warnings.warn("angular threshold is out of range: {}".format(action))
            action = np.clip(a=action, a_min=self.action_threshold_space.low, a_max=self.action_threshold_space.high)
        return action

    def _get_robot_state(self):
        self.current_imu = np.array(copy.deepcopy(self.current_imu_raw))

        if self.n_step == 0:
            self.first_pose = np.concatenate((copy.deepcopy(self.current_position_raw),
                                              copy.deepcopy(self.current_orientation_raw)), axis=-1)
        pose = np.concatenate((copy.deepcopy(self.current_position_raw),
                               copy.deepcopy(self.current_orientation_raw)), axis=-1)
        self.current_pose = self._relative_pose_to_first_frame(pose)

        self.current_vel = np.array(copy.deepcopy(self.current_vel_raw))

    def _get_lidar_map_observation(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._get_lidar(), self._get_elevation_map()

    def _check_status(self):
        roll = abs(self.current_pose[3]) if abs(self.current_pose[3]) < math.pi else abs(self.current_pose[3]) - math.pi
        pitch = abs(self.current_pose[4]) if abs(self.current_pose[4]) < math.pi else abs(
            self.current_pose[4]) - math.pi
        if self.relative_goal[0] <= self.successful_threshold:
            self.robot_terminate = True
            print("terminated because successful")
        elif self.lidar_min_dis < self.collision_threshold:
            self.robot_terminate = True
            print("terminated because in collision {}".format(self.lidar_min_dis))
        elif roll > self.stable_threshold or pitch > self.stable_threshold:
            self.robot_terminate = True
            print("terminated because unstable roll:{:.3f}, pitch:{:.3f}".format(roll * 180. / math.pi,
                                                                                 pitch * 180. / math.pi))

    def _test_state(self):
        assert isinstance(self.current_pose, np.ndarray), "self.current_pose need to be an np array"
        assert self.current_pose.shape == (6,), "self.current_pose has a wrong shape {}".format(self.current_pose.shape)
        assert isinstance(self.goal, np.ndarray), "self.goal need to be an np array"
        assert self.goal.shape == (3,), "self.goal has a wrong shape {}".format(self.goal.shape)
        assert isinstance(self.current_imu, np.ndarray), "self.current_imu need to be an np array"
        assert self.current_imu.shape == (6,), "self.current_imu has a wrong shape {}".format(self.current_imu.shape)
        assert isinstance(self.current_vel, np.ndarray), "self.current_vel need to be an np array"
        assert self.current_vel.shape == (6,), "self.current_vel has a wrong shape {}".format(self.current_vel.shape)

    def _test_observation(self):
        assert isinstance(self.current_lidar, np.ndarray), "self.current_lidar need to be an np array"
        assert self.current_lidar.shape == lidar_shape, \
            "self.current_lidar has a wrong shape {}".format(self.current_lidar.shape)
        assert isinstance(self.current_elevation_map, np.ndarray), "self.current_elevation_map need to be an np array"
        assert self.current_elevation_map.shape == elevation_map_shape, \
            "self.current_elevation_map has a wrong shape {}".format(self.current_elevation_map.shape)
