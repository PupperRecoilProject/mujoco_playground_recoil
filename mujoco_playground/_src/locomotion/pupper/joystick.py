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
"""Joystick task for Pupper."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
# Step 1: Import Pupper's base and constants
from mujoco_playground._src.locomotion.pupper import base as pupper_base
from mujoco_playground._src.locomotion.pupper import pupper_constants as consts


def default_config() -> config_dict.ConfigDict:
  """Gets the default configuration for the Pupper joystick task."""
  config = config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,

      # cascade control
      cascade_pos_kp = consts.CASCADE_POS_KP,
      cascade_vel_kp = consts.CASCADE_VEL_KP,
      cascade_max_target_velocity_rad_s = consts.CASCADE_MAX_TARGET_VELOCITY_RAD_S,

      action_repeat=1,
      action_scale=0.5, #0.5
      history_len=1, # This seems to be unused in the original code
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              tracking_lin_vel=3.0, # 1.0
              tracking_ang_vel=1.5, # 0.5
              lin_vel_z=-0.5,
              ang_vel_xy=-0.05,
              orientation=-5.0,
              dof_pos_limits=-1.0,
              pose=0.1,             # 0.5
              termination=-1.0,
              stand_still=-1.0,
              torques=-0.0002,
              action_rate=-0.01,
              energy=-0.001,
              feet_clearance=-2.0,
              feet_height=-0.2,
              feet_slip=-0.1,
              feet_air_time=0.1,
          ),
          tracking_sigma=0.25,
          # Step 2: Adjust max_foot_height for the shorter Pupper
          max_foot_height=0.06, # Go1 was 0.1, Pupper legs are shorter
      ),
      pert_config=config_dict.create(
          enable=False,  # False
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      command_config=config_dict.create(
          a=[0.4, 0.7, 0.4], # Reduced command range for smaller Pupper # a=[1.0, 0.5, 0.8]
          b=[0.25, 0.9, 0.5],# b=[0.9, 0.25, 0.5]
      ),
  )
  return config


# Step 3: Rename class and inherit from PupperEnv
class Joystick(pupper_base.PupperEnv):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    """Initializes task-specific parameters."""
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # This logic is general and works for Pupper
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    # Use ROOT_BODY from constants file for robustness
    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )
    
    # This sensor address logic is general and correct
    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)

    # === 新增代碼: 從配置中讀取並保存級聯控制器增益 ===
    self.pos_kp = self._config.cascade_pos_kp
    self.vel_kp = self._config.cascade_vel_kp
    self.max_target_vel = self._config.cascade_max_target_velocity_rad_s
    # === 修改結束 ===

  # `reset` and `step` methods are complex but highly general.
  # The core logic of updating state, commands, and perturbations
  # is independent of the robot's morphology, so we keep them.
  
  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # Initial randomization of position and orientation
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-np.pi, maxval=np.pi)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # Initial randomization of velocity
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=self._default_pose)

    # Initialize perturbation and command scheduling
    rng, key1, key2, key3 = jax.random.split(rng, 4)
    time_until_next_pert = jax.random.uniform(key1, minval=self._config.pert_config.kick_wait_times[0], maxval=self._config.pert_config.kick_wait_times[1])
    steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(jp.int32)
    pert_duration_seconds = jax.random.uniform(key2, minval=self._config.pert_config.kick_durations[0], maxval=self._config.pert_config.kick_durations[1])
    pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(jp.int32)
    pert_mag = jax.random.uniform(key3, minval=self._config.pert_config.velocity_kick[0], maxval=self._config.pert_config.velocity_kick[1])

    rng, key1, key2 = jax.random.split(rng, 3)
    time_until_next_cmd = jax.random.exponential(key1) * 5.0
    steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(jp.int32)
    cmd = self.sample_command(key2, jp.zeros(3)) # Start with a random command

    info = {
        "rng": rng,
        "command": cmd,
        "steps_until_next_cmd": steps_until_next_cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    
    # 1. 得到 RL 策略輸出的目標角度 (邏輯不變)
    target_q = self._default_pose + action * self._config.action_scale

    # 2. 計算關節端的最大力矩
    # 計算對應的最大電流 (A)，用於飽和
    max_motor_current_A = consts.MAX_MOTOR_TORQUE / consts.TORQUE_CONSTANT # 約 1.0 / 0.333 = 3A

    # 3. 在循環中執行級聯控制和模擬步驟
    def cascade_control_step(i, data):
        # 讀取當前關節狀態
        current_q = data.qpos[7:]
        current_v = data.qvel[6:]

        # === 級聯控制邏輯 (與您的 Teensy 完全一樣) ===
        
        # --- 外環: 位置控制器 (P-Controller) ---
        pos_error = target_q - current_q
        # 計算目標速度
        target_v = self.pos_kp * pos_error
        # 限制目標速度
        target_v = jp.clip(target_v, 
                           -self.max_target_vel, 
                           self.max_target_vel)

        # --- 內環: 速度控制器 (P-Controller) ---
        vel_error = target_v - current_v
        # 計算目標電流 (單位: A)
        target_current = self.vel_kp * vel_error

        # --- 電流飽和 ---
        # 限制電流在物理範圍內
        target_current = jp.clip(target_current, -max_motor_current_A, max_motor_current_A)
        
        # === 邏輯結束 ===

        # 將目標電流直接作為控制信號發送給 <general> 致動器
        # 因為我們的 XML 中 gain 是 Kt，ctrlrange 是電流範圍，所以這裡可以直接用
        final_ctrl = target_current
        
        # 應用控制信號並執行一步模擬
        data = data.replace(ctrl=final_ctrl)
        data = mjx.step(self.mjx_model, data)
        return data

    # 執行 n_substeps 次高頻控制
    data = jax.lax.fori_loop(0, self.n_substeps, cascade_control_step, state.data)
    # --- 修改結束 ---

    # Contact detection and foot state tracking
    contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info, state.metrics, done, first_contact, contact)
    rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
    
    # Bookkeeping for next step
    info = state.info
    info["last_last_act"] = info["last_act"]
    info["last_act"] = action
    info["steps_until_next_cmd"] -= 1
    info["rng"], key1, key2 = jax.random.split(info["rng"], 3)
    info["command"] = jp.where(
        info["steps_until_next_cmd"] <= 0,
        self.sample_command(key1, info["command"]),
        info["command"],
    )
    info["steps_until_next_cmd"] = jp.where(
        done | (info["steps_until_next_cmd"] <= 0),
        jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
        info["steps_until_next_cmd"],
    )
    info["feet_air_time"] *= ~contact
    info["last_contact"] = contact
    info["swing_peak"] *= ~contact
    
    metrics = state.metrics
    for k, v in rewards.items():
      metrics[f"reward/{k}"] = v
    metrics["swing_peak"] = jp.mean(info["swing_peak"])

    done = done.astype(reward.dtype)
    return state.replace(data=data, obs=obs, reward=reward, done=done, info=info, metrics=metrics)
  
  # _get_termination, _get_obs, and reward functions are mostly general.
  # The logic depends on physical quantities that are correctly calculated
  # for Pupper due to our consistent XML and constants.
  
  def _get_termination(self, data: mjx.Data) -> jax.Array:
    # Terminate if the robot falls over (torso's z-axis points down)
    return self.get_upvector(data)[-1] < 0.0

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> Dict[str, jax.Array]:
    # Observation construction is general and well-designed. No changes needed.
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (gyro + (2*jax.random.uniform(noise_rng, shape=gyro.shape)-1) * self._config.noise_config.level * self._config.noise_config.scales.gyro)
    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (gravity + (2*jax.random.uniform(noise_rng, shape=gravity.shape)-1) * self._config.noise_config.level * self._config.noise_config.scales.gravity)
    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (joint_angles + (2*jax.random.uniform(noise_rng, shape=joint_angles.shape)-1) * self._config.noise_config.level * self._config.noise_config.scales.joint_pos)
    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (joint_vel + (2*jax.random.uniform(noise_rng, shape=joint_vel.shape)-1) * self._config.noise_config.level * self._config.noise_config.scales.joint_vel)
    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (linvel + (2*jax.random.uniform(noise_rng, shape=linvel.shape)-1) * self._config.noise_config.level * self._config.noise_config.scales.linvel)
    
    state_obs = jp.hstack([
        noisy_linvel,
        noisy_gyro,
        noisy_gravity,
        noisy_joint_angles - self._default_pose,
        noisy_joint_vel,
        info["last_act"],
        info["command"],
    ])

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

    privileged_state = jp.hstack([
        state_obs,
        gyro, accelerometer, gravity, linvel, angvel,
        joint_angles - self._default_pose,
        joint_vel,
        data.actuator_force,
        info["last_contact"],
        feet_vel,
        info["feet_air_time"],
        data.xfrc_applied[self._torso_body_id, :3],
        info["steps_since_last_pert"] >= info["steps_until_next_pert"],
    ])

    return {"state": state_obs, "privileged_state": privileged_state}
    
  # The individual reward functions are all kept, as their logic is sound
  # and general. The only change was `max_foot_height` in the config.

  def _get_reward(self, data: mjx.Data, action: jax.Array, info: dict[str, Any], metrics: dict[str, Any], done: jax.Array, first_contact: jax.Array, contact: jax.Array) -> dict[str, jax.Array]:
    del metrics
    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(info["command"], self.get_local_linvel(data)),
        "tracking_ang_vel": self._reward_tracking_ang_vel(info["command"], self.get_gyro(data)),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_upvector(data)),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "termination": self._cost_termination(done),
        "pose": self._reward_pose(data.qpos[7:]),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data),
        "feet_height": self._cost_feet_height(info["swing_peak"], first_contact, info),
        "feet_air_time": self._reward_feet_air_time(info["feet_air_time"], first_contact, info["command"]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
    }

  def _reward_tracking_lin_vel(self, commands: jax.Array, local_vel: jax.Array) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(self, commands: jax.Array, ang_vel: jax.Array) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    # Penalize non flat base orientation by penalizing the xy components of the up-vector
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array) -> jax.Array:
    del last_last_act
    return jp.sum(jp.square(act - last_act))

  def _reward_pose(self, qpos: jax.Array) -> jax.Array:
    weight = jp.array([1.0, 1.0, 0.1] * 4) # Penalize abduction and hip more than knee
    return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

  def _cost_stand_still(self, commands: jax.Array, qpos: jax.Array) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    # Penalize pose deviation only when command is near zero
    return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.1)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_feet_slip(self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]) -> jax.Array:
    cmd_norm = jp.linalg.norm(info["command"])
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.1)

  def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    # Penalize feet getting too high during swing
    delta = jp.maximum(foot_z - self._config.reward_config.max_foot_height, 0.0)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(self, swing_peak: jax.Array, first_contact: jax.Array, info: dict[str, Any]) -> jax.Array:
    cmd_norm = jp.linalg.norm(info["command"])
    # Penalize if the foot didn't reach a minimum height during swing
    error = swing_peak - 0.02 # e.g., require at least 2cm clearance
    return jp.sum(jp.maximum(-error, 0) * first_contact) * (cmd_norm > 0.1)

  def _reward_feet_air_time(self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    # Reward air time around a target duration (e.g., 0.1s)
    rew_air_time = jp.sum(jp.exp(-100 * jp.square(air_time - 0.1)) * first_contact)
    rew_air_time *= (cmd_norm > 0.1)
    return rew_air_time

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    # This function is general and does not need modification
    def gen_dir(rng: jax.Array) -> jax.Array:
      angle = jax.random.uniform(rng, minval=0.0, maxval=2 * jp.pi)
      return jp.array([jp.cos(angle), jp.sin(angle), 0.0])
    def apply_pert(state: mjx_env.State) -> mjx_env.State:
      t = state.info["pert_steps"] * self.dt
      u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
      force = (u_t * self._torso_mass * state.info["pert_mag"] / state.info["pert_duration_seconds"])
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(force * state.info["pert_dir"])
      data = state.data.replace(xfrc_applied=xfrc_applied)
      info = state.info
      info["steps_since_last_pert"] = jp.where(info["pert_steps"] >= info["pert_duration"], 0, info["steps_since_last_pert"])
      info["pert_steps"] += 1
      return state.replace(data=data, info=info)
    def wait(state: mjx_env.State) -> mjx_env.State:
      info = state.info
      info["rng"], rng = jax.random.split(info["rng"])
      info["steps_since_last_pert"] += 1
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      data = state.data.replace(xfrc_applied=xfrc_applied)
      info["pert_steps"] = jp.where(info["steps_since_last_pert"] >= info["steps_until_next_pert"], 0, info["pert_steps"])
      info["pert_dir"] = jp.where(info["steps_since_last_pert"] >= info["steps_until_next_pert"], gen_dir(rng), info["pert_dir"])
      return state.replace(data=data, info=info)
    return jax.lax.cond(state.info["steps_since_last_pert"] >= state.info["steps_until_next_pert"], apply_pert, wait, state)

  def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
    # This function is general and does not need modification
    rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
    y_k = jax.random.uniform(y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a)
    z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
    w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
    return x_k - w_k * (x_k - y_k * z_k)