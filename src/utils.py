from collections import deque
from typing import Dict, Union
import numpy as np
import torch as th
import random
import copy

from gym.spaces import Dict as DictSapce
from gym import Space

TensorDict = Dict[Union[str, int], th.Tensor]


def obs_as_tensor(obs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]], device: th.device, en_copy=False,
                  observation_space: Union[Space, DictSapce] = None) -> Union[th.Tensor, TensorDict]:
    if isinstance(obs, np.ndarray):
        if en_copy:
            obs = copy.deepcopy(obs)
        if observation_space is not None:
            obs = np.array(obs).reshape((-1,) + observation_space.shape)
        return th.as_tensor(np.array(obs, dtype=np.float32)).to(device)
    elif isinstance(obs, dict):
        observation = {}
        for (key, _obs) in obs.items():
            if en_copy:
                _obs = copy.deepcopy(_obs)
            if observation_space is not None:
                _obs = np.array(_obs).reshape((-1,) + observation_space[key].shape)
            observation[key] = th.as_tensor(np.array(_obs, dtype=np.float32)).to(device)
        return observation
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")


def get_device(device: Union[th.device, str] = "cuda") -> th.device:
    if isinstance(device, str):
        assert device == "cuda" or device == "cuda:0" or device == "cuda:1" or device == "cpu", \
            "device should only be 'cuda' or 'cpu' "
    device = th.device(device)
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")
    return device


def safe_mean(arr: Union[np.ndarray, list, deque]) -> np.ndarray:
    return np.nan if len(arr) == 0 else np.mean(arr)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Returns 1 - Var[y-ypred] / Var[y]
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
