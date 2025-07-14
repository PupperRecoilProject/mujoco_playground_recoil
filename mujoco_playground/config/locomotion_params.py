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
"""RL config for Locomotion envs."""

from ml_collections import config_dict

from mujoco_playground._src import locomotion


def brax_ppo_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax PPO config for the given environment."""
  env_config = locomotion.get_default_config(env_name)

  # --- Base RL config, used by default unless overridden below ---
  rl_config = config_dict.create(
      num_timesteps=100_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=20,
      num_minibatches=32,
      num_updates_per_batch=4,
      discounting=0.97,
      learning_rate=3e-4,
      entropy_cost=1e-2,
      num_envs=8192,
      batch_size=256,
      max_grad_norm=1.0,
      network_factory=config_dict.create(
          policy_hidden_layer_sizes=(128, 128, 128, 128),
          value_hidden_layer_sizes=(256, 256, 256, 256, 256),
          policy_obs_key="state",
          value_obs_key="state",
      ),
  )

  # --- Specializations for different environments ---

  if env_name in ("Go1JoystickFlatTerrain", "Go1JoystickRoughTerrain"):
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 10
    rl_config.num_resets_per_eval = 1
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in ("Go1Handstand", "Go1Footstand"):
    rl_config.num_timesteps = 100_000_000
    rl_config.num_evals = 5
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name == "Go1Backflip":
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 10
    rl_config.discounting = 0.95
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name == "Go1Getup":
    rl_config.num_timesteps = 50_000_000
    rl_config.num_evals = 5
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  # ==============================================================================
  # START OF PUPPER CONFIGURATION
  # ==============================================================================

  elif env_name in ("PupperJoystickFlatTerrain", "PupperJoystickRoughTerrain", "PupperJoystickWithGun"):
    # We copy the Go1Joystick config as a starting point.
    # Pupper is smaller and lighter, so might need fewer timesteps.
    rl_config.num_timesteps = 100_000_000 # Reduced from Go1's 200M
    rl_config.num_evals = 20
    rl_config.num_resets_per_eval = 1
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state", # Using privileged state is a good idea
    )

  elif env_name == "PupperGetup":
    # We copy the Go1Getup config. This task is simpler.
    rl_config.num_timesteps = 50_000_000 # Reduced from Go1's 50M
    rl_config.num_evals = 5
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  # ==============================================================================
  # END OF PUPPER CONFIGURATION
  # ==============================================================================

  # --- (Keep the rest of the original elif blocks for other robots) ---

  elif env_name in ("G1JoystickFlatTerrain", "G1JoystickRoughTerrain"):
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 20
    rl_config.clipping_epsilon = 0.2
    rl_config.num_resets_per_eval = 1
    rl_config.entropy_cost = 0.005
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in (
      "BerkeleyHumanoidJoystickFlatTerrain",
      "BerkeleyHumanoidJoystickRoughTerrain",
  ):
    rl_config.num_timesteps = 150_000_000
    rl_config.num_evals = 15
    rl_config.clipping_epsilon = 0.2
    rl_config.num_resets_per_eval = 1
    rl_config.entropy_cost = 0.005
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in (
      "T1JoystickFlatTerrain",
      "T1JoystickRoughTerrain",
  ):
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 20
    rl_config.clipping_epsilon = 0.2
    rl_config.num_resets_per_eval = 1
    rl_config.entropy_cost = 0.005
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in ("ApolloJoystickFlatTerrain",):
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 20
    rl_config.clipping_epsilon = 0.2
    rl_config.num_resets_per_eval = 1
    rl_config.entropy_cost = 0.005
    rl_config.network_factory = config_dict.create(
      policy_hidden_layer_sizes=(512, 256, 128),
      value_hidden_layer_sizes=(512, 256, 128),
      policy_obs_key="state",
      value_obs_key="privileged_state",
    )

  elif env_name in (
      "BarkourJoystick",
      "H1InplaceGaitTracking",
      "H1JoystickGaitTracking",
      "Op3Joystick",
      "SpotFlatTerrainJoystick",
      "SpotGetup",
      "SpotJoystickGaitTracking",
  ):
    pass  # use default config
  else:
    raise ValueError(f"Unsupported env: {env_name}")

  return rl_config


# --- (The rsl_rl_config function remains unchanged) ---

def rsl_rl_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned RSL-RL PPO config for the given environment."""

  rl_config = config_dict.create(
      seed=1,
      runner_class_name="OnPolicyRunner",
      policy=config_dict.create(
          init_noise_std=1.0,
          actor_hidden_dims=[512, 256, 128],
          critic_hidden_dims=[512, 256, 128],
          # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
          activation="elu",
          class_name="ActorCritic",
      ),
      algorithm=config_dict.create(
          class_name="PPO",
          value_loss_coef=1.0,
          use_clipped_value_loss=True,
          clip_param=0.2,
          entropy_coef=0.001,
          num_learning_epochs=5,
          # mini batch size = num_envs*nsteps / nminibatches
          num_mini_batches=4,
          learning_rate=3.0e-4,  # 5.e-4
          schedule="fixed",  # could be adaptive, fixed
          gamma=0.99,
          lam=0.95,
          desired_kl=0.01,
          max_grad_norm=1.0,
      ),
      num_steps_per_env=24,  # per iteration
      max_iterations=100000,  # number of policy updates
      empirical_normalization=True,
      # logging
      save_interval=50,  # check for potential saves every this many iterations
      experiment_name="test",
      run_name="",
      # load and resume
      resume=False,
      load_run="-1",  # -1 = last run
      checkpoint=-1,  # -1 = last saved model
      resume_path=None,  # updated from load_run and chkpt
  )

  if env_name in (
      "Go1Getup",
      "BerkeleyHumanoidJoystickFlatTerrain",
      "G1Joystick",
      "Go1JoystickFlatTerrain",
      "PupperGetup", # Also add Pupper here for shorter RSL-RL runs if needed
      "PupperJoystickFlatTerrain",
  ):
    rl_config.max_iterations = 1000
  if env_name == "Go1JoystickFlatTerrain":
    rl_config.algorithm.learning_rate = 3e-4
    rl_config.algorithm.schedule = "fixed"

  return rl_config


# ==============================================================================
# START: BRAX SAC CONFIGURATION
# ==============================================================================
def brax_sac_config(env_name: str) -> config_dict.ConfigDict:
  """
  Returns a tuned Brax SAC config for the given locomotion environment.
  This function assumes the environment has been prepared for flat observations
  (e.g., PupperJoystickSac environments).
  """
  # 從 locomotion 模組獲取環境的默認配置
  # 這將提供 episode_length 等基本資訊
  env_config = locomotion.get_default_config(env_name)

  # --- 基礎 SAC RL 配置 ---
  # 這些參數是 SAC 演算法的標準起點
  rl_config = config_dict.create(
      # 訓練流程參數
      num_timesteps=16_777_216,  # SAC 數據效率高，總步數可相對較少
      num_evals=64,  # 評估頻率
      episode_length=env_config.episode_length,
      action_repeat=1,
      num_envs=8192,  # SAC 通常使用多個並行環境來收集數據到 Replay Buffer
      num_eval_envs=128,
      seed=0,

      # Replay Buffer 參數
      min_replay_size=16_384, # 學習開始前的探索步數
      max_replay_size=1_048_576,
      batch_size=512,
      grad_updates_per_step=64, # 每收集一步數據，進行一次網路更新

      # SAC 演算法核心參數
      learning_rate=3e-4,
      discounting=0.99,
      reward_scaling=5.0,  # SAC 對獎勵尺度敏感
      tau=0.005,  # 目標網路軟更新係數
      normalize_observations=True, # 【關鍵】現在可以安全地開啟！
      deterministic_eval=True,  # 評估時使用確定性策略

      # 網路結構參數 (將由 network_factory 處理)
      network_factory=config_dict.create(
          # SAC 的 Actor 和 Critic 網路通常是對稱的、深度適中
          hidden_layer_sizes=(256, 256),
          # 我們假設 make_sac_networks 能夠從 observation_spec 中處理 'state'
          # 如果您需要特權觀測，且環境提供，可以在此指定
          # 例如: policy_obs_key='state', value_obs_key='privileged_state',
      ),
  )

  # --- 針對不同環境的特殊化配置 ---
  # 這裡只針對我們新創建的 PupperJoystickSac 環境進行配置
  if env_name in ("PupperJoystickSacFlatTerrain", "PupperJoystickSacRoughTerrain"):
    # 這些任務相對複雜，需要更多的訓練和更大的網路容量
    rl_config.num_timesteps = 100_000_000
    rl_config.grad_updates_per_step = 32 # 增加數據利用率
    rl_config.reward_scaling = 20.0 # 增強獎勵信號
    rl_config.network_factory = config_dict.create(
        hidden_layer_sizes=(512, 256, 128), # 更大容量的網路
        # 如果需要，這裡可以指定 obs_key
        # policy_obs_key='state',
        # value_obs_key='privileged_state',
    )
  
  # 如果您有其他 SAC 訓練的環境，可以在這裡添加 elif 塊

  return rl_config

# ==============================================================================
# END: BRAX SAC CONFIGURATION
# ==============================================================================
