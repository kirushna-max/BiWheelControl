from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict


import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

path = '../assets/urdf/Bipedal.xml'

mj_model = mujoco.MjModel.from_xml_path(path)
mj_data = mujoco.MjData(mj_model)
# Renderer creation needs a working GLFW/OpenGL display.  MJX itself does not
# need one, and this script does not currently render a frame.  Set
# ENABLE_MUJOCO_RENDERER=1 when running in a desktop session where rendering
# is required.
renderer = (
    mujoco.Renderer(mj_model)
    if os.environ.get("ENABLE_MUJOCO_RENDERER") == "1"
    else None
)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)
print(mj_data.qpos, type(mj_data.qpos))
print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())
