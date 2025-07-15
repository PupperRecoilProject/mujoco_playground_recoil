# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fall recovery task for the Pupper."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
# Step 1: Import Pupper's base and constants
from mujoco_playground._src.locomotion.pupper import base as pupper_base
from mujoco_playground._src.locomotion.pupper import pupper_constants as consts


def default_config() -> config_dict.ConfigDict:
  """Gets the default configuration for the Pupper getup task."""
  # Most parameters can be inherited from Go1, as they are general RL settings.
  # We will adjust parameters specific to Pupper's morphology later.
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      Kp=consts.MOTOR_KP,  # Use Kp from our constants
      Kd=consts.MOTOR_KD,  # Use Kd from our constants
      episode_length=300,
      drop_from_height_prob=0.6,
      settle_time=0.5,
      action_repeat=1,
      action_scale=0.12, # 0.12
      soft_joint_pos_limit_factor=0.95,
      energy_termination_threshold=np.inf,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              orientation=5.0,
              torso_height=10.0,
              posture=0.1,
              stand_still=0.1,
              action_rate=0,
              dof_pos_limits=-0.1,
              torques=0,  #-1e-6
              dof_acc=0,# -2.5e-7
              dof_vel=0, # -0.1
          ),
      ),
  )


# Step 2: Rename the class and inherit from PupperEnv
class Getup(pupper_base.PupperEnv):
  """Recover from a fall and stand up.

  Observation space:
      - Gyroscope readings (3)
      - Gravity vector (3)
      - Joint angles relative to default pose (12)
      - Joint velocities (12)
      - Last action (12)

  Action space: Joint angle displacements (12) added to the current joint angles.

  Reward function:
      - Orientation: The torso should be upright.
      - Torso height: The torso should be at a desired height.
      - Posture: The robot should be in the neutral pose.
      - Stand still: Minimize actions when standing.
      - Regularization: Minimize torques, action rate, etc. for smoother control.
  """

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    # Step 3: Use the Pupper XML file
    # We use task_to_xml to be consistent with the new constants file structure
    xml_path = consts.task_to_xml("flat_terrain").as_posix()
    super().__init__(
        xml_path=xml_path,
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    """Initializes task-specific parameters."""
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # The joint limit logic is general and does not need to be changed.
    # We use [1:] to skip the freejoint, which has no range.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    self._settle_steps = int(self._config.settle_time / self.sim_dt)
    
    # Step 4: CRITICAL - Adjust desired height for Pupper
    # Go1 height was ~0.275m. Pupper is shorter. 
    # From our XML, home height is 0.23m. Let's set a target slightly below that.
    self._z_des = 0.16
    
    self._up_vec = jp.array([0.0, 0.0, -1.0]) # This is incorrect for MuJoCo's gravity
    # Let's correct it. Gravity is -z, so the up vector in world frame is [0, 0, 1].
    # The reward function uses get_gravity, which is in the IMU's frame.
    # The original implementation seems to have a bug. Let's assume the intent
    # is to align the IMU's z-axis with the world's z-axis.
    self._up_vec = jp.array([0.0, 0.0, 1.0]) # Target for IMU's z-axis in its own frame.
    self._imu_site_id = self._mj_model.site("imu").id

  def _get_random_qpos(self, rng: jax.Array) -> jax.Array:
    """Generate an initial configuration where the robot is at a height of 0.5m
    with a random orientation and joint angles."""
    rng, orientation_rng, qpos_rng = jax.random.split(rng, 3)

    qpos = jp.zeros(self.mjx_model.nq)

    # Initialize height and orientation of the root body.
    height = 0.5 # Dropping from 0.5m is fine for both robots.
    qpos = qpos.at[2].set(height)
    quat = jax.random.normal(orientation_rng, (4,))
    quat /= jp.linalg.norm(quat) + 1e-6
    qpos = qpos.at[3:7].set(quat)

    # Randomize joint angles. This logic is general.
    qpos = qpos.at[7:].set(
        jax.random.uniform(
            qpos_rng, (12,), minval=self._lowers, maxval=self._uppers
        )
    )

    return qpos

  # The reset and step methods are highly general and rely on the config
  # and other helper methods. They don't need significant changes.
  
  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Sample a random initial configuration with some probability.
    rng, key1, key2 = jax.random.split(rng, 3)
    qpos = jp.where(
        jax.random.bernoulli(key1, self._config.drop_from_height_prob),
        self._get_random_qpos(key2),
        self._init_q,
    )

    # Sample a random root velocity.
    rng, key = jax.random.split(rng)
    qvel = jp.zeros(self.mjx_model.nv)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

    # Let the robot settle for a few steps.
    data = mjx_env.step(self.mjx_model, data, qpos[7:], self._settle_steps)
    data = data.replace(time=0.0)

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    motor_targets = state.data.qpos[7:] + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info, state.metrics, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    # Bookkeeping.
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = jp.float32(done)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state
    
  # The _get_termination, _get_obs, and reward functions are also general.
  # The key change was `_z_des` which is used in the reward calculations.

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    energy = jp.sum(jp.abs(data.actuator_force * data.qvel[6:]))
    energy_termination = energy > self._config.energy_termination_threshold
    return energy_termination

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    # This observation function is well-designed and general.
    # It uses relative joint angles (noisy_joint_angles - self._default_pose),
    # which is great practice. No changes needed.
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    state = jp.concatenate([
        noisy_gyro,
        noisy_gravity,
        noisy_joint_angles - self._default_pose,
        noisy_joint_vel,
        info["last_act"],
    ])
    
    # Privileged state construction is also general.
    accelerometer = self.get_accelerometer(data)
    linvel = self.get_local_linvel(data)
    angvel = self.get_global_angvel(data)
    torso_height = data.site_xpos[self._imu_site_id][2]

    privileged_state = jp.hstack([
        state,
        gyro,
        accelerometer,
        linvel,
        angvel,
        joint_angles,
        joint_vel,
        data.actuator_force,
        torso_height,
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    del done, metrics  # Unused.

    torso_height = data.site_xpos[self._imu_site_id][2]
    joint_angles = data.qpos[7:]
    joint_torques = data.actuator_force

    # Note: get_gravity() returns gravity vector in local IMU frame.
    # We want the IMU's z-axis to point opposite to the world gravity vector.
    # The world gravity is (0,0,-g). IMU's z-axis in world frame is data.site_xmat[imu_id][:, 2].
    # So we want data.site_xmat[imu_id][:, 2] to be close to [0, 0, 1].
    # get_gravity() is `imu_rot_matrix.T @ [0, 0, -1]`. We want this to be [0, 0, -1].
    # So the orientation reward logic is slightly confusing but might work.
    # Let's use `upvector` sensor instead for clarity if available, or stick to the original logic.
    # `get_upvector` gives site_xmat[2], which is the z-axis of the site in world frame.
    up_vec_in_world = self.get_upvector(data)

    is_upright = self._is_upright(up_vec_in_world)
    is_at_desired_height = self._is_at_desired_height(torso_height)
    gate = is_upright * is_at_desired_height

    return {
        "orientation": self._reward_orientation(up_vec_in_world),
        "torso_height": self._reward_height(torso_height),
        "posture": self._reward_posture(joint_angles, is_upright),
        "stand_still": self._reward_stand_still(action, gate),
        "action_rate": self._cost_action_rate(action, info),
        "torques": self._cost_torques(joint_torques),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),
        "dof_vel": self._cost_dof_vel(data.qvel[6:]),
    }

  # All individual reward/cost functions below are general and do not need changes.
  # They operate on physical quantities whose meaning is the same for Pupper.

  def _is_upright(self, up_vec_in_world: jax.Array, ori_tol: float = 0.01) -> jax.Array:
    # Reward for the up-vector of the torso pointing towards the world z-axis.
    ori_error = jp.sum(jp.square(self._up_vec - up_vec_in_world))
    return ori_error < ori_tol

  def _is_at_desired_height(
      self, torso_height: jax.Array, pos_tol: float = 0.005
  ) -> jax.Array:
    height_error = jp.abs(self._z_des - torso_height)
    return height_error < pos_tol

  def _reward_orientation(self, up_vec_in_world: jax.Array) -> jax.Array:
    error = jp.sum(jp.square(self._up_vec - up_vec_in_world))
    return jp.exp(-2.0 * error)

  def _reward_height(self, torso_height: jax.Array) -> jax.Array:
    # Encourage torso height to be close to the desired height `_z_des`.
    error = jp.square(torso_height - self._z_des)
    return jp.exp(-5.0 * error) # Using a squared exponential for a peak at z_des

  def _reward_posture(
      self, joint_angles: jax.Array, gate: jax.Array
  ) -> jax.Array:
    cost = jp.sum(jp.square(joint_angles - self._default_pose))
    rew = jp.exp(-0.5 * cost)
    return gate * rew

  def _reward_stand_still(self, act: jax.Array, gate: jax.Array) -> jax.Array:
    cost = jp.sum(jp.square(act))
    rew = jp.exp(-0.5 * cost)
    return gate * rew

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_action_rate(
      self, act: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    c1 = jp.sum(jp.square(act - info["last_act"]))
    c2 = jp.sum(jp.square(act - 2 * info["last_act"] + info["last_last_act"]))
    return c1 + c2

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_dof_vel(self, qvel: jax.Array) -> jax.Array:
    max_velocity = 2.0 * jp.pi  # rad/s
    cost = jp.maximum(jp.abs(qvel) - max_velocity, 0.0)
    return jp.sum(jp.square(cost))

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))