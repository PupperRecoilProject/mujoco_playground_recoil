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
"""Defines Stanford Pupper quadruped constants."""

import numpy as np
from etils import epath

from mujoco_playground._src import mjx_env

# ==================== FILE PATHS ====================
# 將路徑指向 pupper 資料夾
ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "pupper"

# 假設您只創建了我們之前討論的 'scene_mjx.xml'
# 我們可以將所有任務都指向這個檔案，或者為未來擴展預留位置
# 這裡我們只定義一個基礎的平地場景
PUPPER_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx.xml"
PUPPER_TERRAIN_SCENE_XML = ROOT_PATH / "xmls" / "scene_mjx.xml"

def task_to_xml(task_name: str) -> epath.Path:
  """Maps a task name to a MuJoCo XML file path."""
  # 您可以根據未來需求擴展這個字典，例如添加 "rough_terrain"
  return {
      # 平地場景訓練
      "flat_terrain": PUPPER_FLAT_TERRAIN_XML,
      #  hfield 場景 (用於地形訓練)
      "terrain_scene": PUPPER_TERRAIN_SCENE_XML,
  }[task_name]


# ==================== BODY AND SENSOR NAMES ====================
# 這些名稱必須與您的 `pupper_mjx.xml` 檔案完全匹配

# 腳部的 site 名稱，用於感測器
FEET_SITES = [
    "FR",
    "FL",
    "RR",
    "RL",
]

# 腳部的 geom 名稱，用於碰撞檢測
FEET_GEOMS = [
    "FR",
    "FL",
    "RR",
    "RL",
]

# 腳部位置感測器的名稱 (基於 XML 中的定義)
FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

# 機器人主軀幹的名稱 (我們在XML中設定為 'torso')
ROOT_BODY = "torso"

# 核心IMU感測器的名稱 (這些都是從Go1的XML複製過來的，所以在我們的Pupper XML中也存在)
UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

# ==================== ROBOT-SPECIFIC CONSTANTS ====================
# 這些是我們之前討論過的、特定於Pupper硬體的常數

ROBOT_NAME = "Pupper"

# 初始關節位置 (qpos)
INIT_QPOS = np.array([
    # FR leg (hip, thigh, calf)
    0.0, 0.4, -0.8,
    # FL leg
    0.0, 0.4, -0.8,
    # RR leg
    0.0, 0.4, -0.8,
    # RL leg
    0.0, 0.4, -0.8,
], dtype=np.float32)

# 關節名稱 (必須與 <actuator> 順序一致)
JOINT_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)

# 預設站立姿態
DEFAULT_ABDUCTION_ANGLE = 0.0
DEFAULT_HIP_ANGLE = 0.4
DEFAULT_KNEE_ANGLE = -0.8

# 新的級聯控制器增益 (與您的 Teensy 代碼匹配)
CASCADE_POS_KP = 16.0  # 外環位置 P 增益 (16.0)
CASCADE_VEL_KP_mA = 500   # 內環速度 P 增益 (500.0)
# 外環輸出的最大目標速度 (rad/s)
CASCADE_MAX_TARGET_VELOCITY_RAD_S = 8.0 # (8.0)

# 馬達物理參數 (從你的 XML 和規格書中獲取)
TORQUE_CONSTANT = 3000  # 單位: N·m / A 
MAX_MOTOR_TORQUE = 1.6   # 馬達本身的最大持續扭矩 (N·m)
# 最大持續電流，單位是 mA
# I(mA) = Torque(N·m) / Kt(N·m/mA) = 1.6 / 0.000333 ≈ 4800 mA
# MAX_MOTOR_CURRENT_mA = 4800.0 # 單位: mA
# --- 修改結束 ---

# 動作空間限制 (從 XML 的 joint range 獲取)
ACTION_LIMITS = np.array([
    # FR leg
    [-1.0472, 1.0472], [-0.76166, 3.81442], [-0.78540, 1.65806],
    # FL leg
    [-1.0472, 1.0472], [-0.76166, 3.81442], [-0.78540, 1.65806],
    # RR leg
    [-1.0472, 1.0472], [-0.76166, 3.81442], [-0.78540, 1.65806],
    # RL leg
    [-1.0472, 1.0472], [-0.76166, 3.81442], [-0.78540, 1.65806],
], dtype=np.float32)

# 自由度數量
NUM_MOTORS = len(JOINT_NAMES)
ACTION_DIM = NUM_MOTORS