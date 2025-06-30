# pupper_constants.py

# Copyright 2024 [您的名字或組織]
# ... (授權資訊保持不變) ...
# ==============================================================================
"""Defines Pupper quadruped constants and asset paths for a flat directory structure."""

from etils import epath
import os

# --- 核心路徑定義 ---
# 自動將根路徑設定為此檔案所在的目錄
# 這使得腳本在任何地方被導入時，都能正確地定位到 'train2.0' 資料夾
PUPPER_ROOT_PATH = epath.Path(os.path.dirname(os.path.abspath(__file__)))

# --- XML 檔案路徑 ---
# 直接指向與此常數檔案位於同一目錄下的 XML 檔案
# 根據您的截圖，您的檔案名中包含 '2.0'
PUPPER_MJX_XML = PUPPER_ROOT_PATH / "pupper_mjx2.0.xml"
SCENE_MJX_XML = PUPPER_ROOT_PATH / "scene_mjx2.0.xml"

# 您可以定義一個預設使用的模型
DEFAULT_XML = SCENE_MJX_XML

def task_to_xml_path(task_name: str) -> epath.Path:
  """根據任務名稱返回對應的 XML 檔案路徑。"""
  # 這個函式現在可以根據您的需求進行擴展
  if task_name == "pupper_direct":
      return PUPPER_MJX_XML
  elif task_name == "scene_default":
      return SCENE_MJX_XML
  else: # 預設情況
      return DEFAULT_XML

# --- 模型元素命名 ---
# 這些名稱應與您 XML 檔案中的定義嚴格匹配

ROOT_BODY = "torso"

FEET_SITES = [
    "foot_front_right",
    "foot_front_left",
    "foot_hind_right",
    "foot_hind_left",
]

FEET_GEOMS = [
    "foot_front_right_collision",
    "foot_front_left_collision",
    "foot_hind_right_collision",
    "foot_hind_left_collision",
]

JOINT_NAMES = [
    "abduction_front_right", "hip_front_right", "knee_front_right",
    "abduction_front_left",  "hip_front_left",  "knee_front_left",
    "abduction_hind_right",  "hip_hind_right",  "knee_hind_right",
    "abduction_hind_left",   "hip_hind_left",   "knee_hind_left",
]

# --- 感測器名稱 (Sensor Names) ---
# 這些名稱必須與您在 XML 的 <sensor> 區塊中定義的 name 完全匹配

# 標準觀測 (Actor 可用)
JOINT_POS_SENSOR = "joint_pos"
IMU_ACCEL_SENSOR = "imu_accel"
IMU_GYRO_SENSOR = "imu_gyro"

# 特權觀測 (Critic 可用)
TORSO_QUAT_SENSOR = "torso_quat"
TORSO_POS_SENSOR = "torso_pos"
TORSO_LIN_VEL_SENSOR = "torso_lin_vel"
TORSO_ANG_VEL_SENSOR = "torso_ang_vel"
JOINT_VEL_SENSOR = "joint_vel"
ACTUATOR_FORCES_SENSOR = "actuator_forces"
FOOT_CONTACTS_SENSOR = "foot_contacts"
SUBTREE_COM_SENSOR = "subtree_com"