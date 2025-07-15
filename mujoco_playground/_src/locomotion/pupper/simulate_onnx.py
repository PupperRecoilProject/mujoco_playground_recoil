# ===================================================================
#      交互式 ONNX 模擬腳本 (最終完整修正版)
# ===================================================================
import mujoco
import numpy as np
import onnxruntime as ort
import time
from collections import deque
import glfw
import sys
from termcolor import cprint

# =======================================================
# ===           1. 檔案路徑與核心設定                 ===
# =======================================================
XML_FILE = 'xmls/scene_mjx.xml'
ONNX_MODEL_PATH = "pupper_ppo_policy_30965760_tf_converted.onnx"
NUM_MOTORS = 12

OBSERVATION_RECIPES = {
    48: [
        'linear_velocity', 'angular_velocity', 'gravity_vector',
        'joint_positions', 'joint_velocities', 'last_action', 'commands',
    ],
}

PHYSICS_TIMESTEP = 0.004
CONTROL_FREQ = 50.0
CONTROL_DT = 1.0 / CONTROL_FREQ
WARMUP_DURATION = 3.0

VELOCITY_ADJUST_STEP = 0.1

# =======================================================
# ===        2. 全局變數和輔助類定義                  ===
# =======================================================

class TuningParams:
    def __init__(self):
        self.kp = 4.0
        self.kd = 0.4
        self.action_scale = 0.5
        self.bias = 0.0

params = TuningParams()
current_command = np.zeros(3, dtype=np.float32)
shared_data = {}

class OnnxPolicy:
    def __init__(self, onnx_path: str):
        cprint(f"Loading ONNX model from: {onnx_path}", "cyan")
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        cprint(f"  - ONNX model loaded. Input name: '{self.input_name}'", "green")

    def get_action(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        onnx_inputs = {self.input_name: obs_dict['state']}
        return self.session.run([self.output_name], onnx_inputs)[0]

class ObservationBuilder:
    def __init__(self, recipe, data, model, default_pose):
        self.recipe = recipe
        self.data = data
        self.model = model
        self.default_pose = default_pose
        self._component_generators = self._register_components()
        for component in self.recipe:
            if component not in self._component_generators:
                sys.exit(f"Error: Obs component '{component}' not in recipe.")

    def _register_components(self):
        return {
            'linear_velocity': self._get_linear_velocity,
            'angular_velocity': self._get_angular_velocity,
            'gravity_vector': self._get_gravity_vector,
            'joint_positions': self._get_joint_positions,
            'joint_velocities': self._get_joint_velocities,
            'last_action': self._get_last_action,
            'commands': self._get_commands,
        }

    def get_observation(self, command, last_action) -> np.ndarray:
        obs_list = [
            self._component_generators[name](command=command, last_action=last_action)
            for name in self.recipe
        ]
        return np.concatenate(obs_list).astype(np.float32)

    def _get_linear_velocity(self, **kwargs): return self.data.sensor('local_linvel').data.copy()
    def _get_angular_velocity(self, **kwargs): return self.data.sensor('gyro').data.copy()
    def _get_gravity_vector(self, **kwargs): return self.data.sensor('accelerometer').data.copy()
    def _get_joint_positions(self, **kwargs): return self.data.qpos[7:] - self.default_pose
    def _get_joint_velocities(self, **kwargs): return self.data.qvel[6:].copy()
    def _get_last_action(self, last_action, **kwargs): return last_action
    def _get_commands(self, command, **kwargs): return command

class DebugOverlay:
    def __init__(self, recipe):
        self.data = {}
        self.recipe = recipe
        self.component_dims = self._calculate_component_dims()

    def _calculate_component_dims(self):
        dims = {'linear_velocity': 3, 'angular_velocity': 3, 'gravity_vector': 3,
                'joint_positions': 12, 'joint_velocities': 12, 'last_action': 12, 'commands': 3}
        return {key: dims[key] for key in self.recipe if key in dims}

    def update(self, **kwargs):
        self.data.update(kwargs)
    
    def render(self, viewport, context):
        if not self.data: return
        help_text = ("--- CONTROLS ---\n\n[Keyboard]\n  WASD/QE: Move/Turn\n  R: Reset\n"
                     "  C: Clear Cmd\n  I/K: Kp +/- \n  L/J: Kd +/- \n  Y/H: ActScl +/- \n  ESC: Exit")
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, viewport, help_text, None, context)
        
        def format_vec(name, vec, precision=3):
            if vec is None: return f"{name:<22}: None"
            return f"{name:<22}: " + np.array2string(vec, precision=precision, floatmode='fixed', suppress_small=True, threshold=100)
        
        info_text = (
            f"Mode: {self.data.get('mode_text', 'N/A')} (Time: {self.data.get('sim_time', 0):.2f} s)\n"
            f"--- Tuning Params ---\n"
            f"Kp: {self.data.get('kp', 0):.1f} | "
            f"Kd: {self.data.get('kd', 0):.2f} | "
            f"ActScl: {self.data.get('action_scale', 0.0):.3f}\n" # <-- 修正了 f-string
            f"--- Command ---\n"
            f"{format_vec('User Command', self.data.get('command'))}\n"
        )
        
        onnx_input_text = "--- ONNX INPUTS (Breakdown) ---\n"
        onnx_input_vec = self.data.get('onnx_input')
        if onnx_input_vec is not None and self.recipe:
            current_idx = 0
            for component_name in self.recipe:
                dim = self.component_dims.get(component_name, 0)
                if dim > 0:
                    value_slice = onnx_input_vec[current_idx : current_idx + dim]
                    onnx_input_text += format_vec(f"{component_name} [{dim}d]", value_slice, 2) + "\n"
                    current_idx += dim
        
        output_text = (
            f"--- ONNX OUTPUTS & CONTROL ---\n"
            f"{format_vec('ONNX Raw Action', self.data.get('action_raw'))}\n"
            f"{format_vec('Final Motor Ctrl', self.data.get('final_ctrl'))}\n"
        )
        
        full_text = info_text + onnx_input_text + output_text
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, full_text, None, context)

def key_callback(window, key, scancode, action, mods):
    if action != glfw.PRESS: return
    if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(window, 1); return
    if key == glfw.KEY_R: shared_data['reset_requested'] = True; return
    if key == glfw.KEY_C: current_command[:] = 0.0; return

    key_map = {
        glfw.KEY_W: (1, -1), glfw.KEY_S: (1, 1), glfw.KEY_A: (0, 1), glfw.KEY_D: (0, -1),
        glfw.KEY_Q: (2, 1), glfw.KEY_E: (2, -1)
    }
    if key in key_map:
        idx, sign = key_map[key]
        current_command[idx] += VELOCITY_ADJUST_STEP * sign

    param_map = {
        glfw.KEY_I: ('kp', 10.0), glfw.KEY_K: ('kp', -10.0),
        glfw.KEY_L: ('kd', 0.1), glfw.KEY_J: ('kd', -0.1),
        glfw.KEY_Y: ('action_scale', 0.01), glfw.KEY_H: ('action_scale', -0.01)
    }
    if key in param_map:
        attr, delta = param_map[key]
        setattr(params, attr, getattr(params, attr) + delta)
    
    params.kp = max(0, params.kp)
    params.kd = max(0, params.kd)
    params.action_scale = max(0, params.action_scale)

# =======================================================
# ===              主模擬程式                      ===
# =======================================================
def main():
    global shared_data, current_command, params

    try:
        model = mujoco.MjModel.from_xml_path(XML_FILE)
        data = mujoco.MjData(model)
        model.opt.timestep = PHYSICS_TIMESTEP
        
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        if torso_id == -1: sys.exit("Error: Body 'torso' not found.")
            
        try:
            home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
            default_pose = model.key_qpos[home_key_id][7:]
            home_pose_actuators = model.key_ctrl[home_key_id].copy()
        except (KeyError, IndexError):
            sys.exit("FATAL ERROR: Keyframe 'home' with valid qpos/ctrl not found in XML.")
            
        policy = OnnxPolicy(ONNX_MODEL_PATH)
        
        model_input_dim = policy.session.get_inputs()[0].shape[1]
        if model_input_dim not in OBSERVATION_RECIPES:
            sys.exit(f"Error: No recipe for input dimension {model_input_dim}.")
        recipe = OBSERVATION_RECIPES[model_input_dim]
        
        obs_builder = ObservationBuilder(recipe, data, model, default_pose)
        last_action = np.zeros(NUM_MOTORS, dtype=np.float32)

        if not glfw.init(): sys.exit("Error: GLFW init failed.")
        window = glfw.create_window(1280, 720, "Interactive ONNX Simulation", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        
        shared_data = {'reset_requested': False}
        glfw.set_window_user_pointer(window, shared_data)
        glfw.set_key_callback(window, key_callback)
        
        cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
        mujoco.mjv_defaultCamera(cam)
        cam.distance, cam.elevation, cam.azimuth = 2.5, -20, 90
        
        scn = mujoco.MjvScene(model, maxgeom=10000)
        con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        debug_overlay = DebugOverlay(recipe)

        # 【關鍵修改】: 重寫 reset_simulation 函數
        def reset_simulation():
            """
            Resets the simulation to a specific keyframe by manually setting
            qpos, qvel, and ctrl.
            """
            cprint("--- Simulation Reset to 'home' state ---", "yellow")
            # 1. 將 data 中的 qpos 和 qvel 設置為 keyframe 中的值
            data.qpos[:] = model.key_qpos[home_key_id]
            data.qvel[:] = model.key_qvel[home_key_id]
            # 2. 如果 keyframe 中有 ctrl，也設置它
            if model.key_ctrl is not None and home_key_id < model.nkey:
                data.ctrl[:] = model.key_ctrl[home_key_id]
            
            # 3. 執行一次正向動力學，以確保所有衍生量（如感測器數據）都已更新
            mujoco.mj_forward(model, data)

            # 4. 重置程式內部狀態
            last_action[:] = 0.0
            shared_data['control_timer'] = data.time
            current_command[:] = 0.0
        
        reset_simulation()
        
        cprint("\n--- Simulation Started ---", "green")
        
        while not glfw.window_should_close(window):
            if shared_data.get('reset_requested'):
                reset_simulation()
                shared_data['reset_requested'] = False

            sim_time = data.time
            control_timer = shared_data.get('control_timer', sim_time)
            
            if control_timer <= sim_time:
                # 應用實時調整的控制器參數
                model.actuator_gainprm[:, 0] = params.kp
                model.dof_damping[6:] = params.kd
                
                # 獲取觀測
                current_obs_dict = obs_builder.get_observation(current_command, last_action)
                onnx_input = current_obs_dict.reshape(1, -1)
                
                action_raw = np.zeros(NUM_MOTORS, dtype=np.float32)
                mode_text = "Warmup"
                final_ctrl = home_pose_actuators # 熱身時保持 home 姿態
                
                if sim_time >= WARMUP_DURATION:
                    mode_text = "ONNX Policy"
                    onnx_action = policy.get_action({'state': onnx_input}).flatten()
                    action_raw = onnx_action # 記錄 ONNX 的原始輸出
                    # 正確地應用動作
                    final_ctrl = home_pose_actuators + action_raw * params.action_scale
                
                last_action[:] = action_raw
                data.ctrl[:] = final_ctrl
                
                shared_data['control_timer'] += CONTROL_DT
                
                debug_overlay.update(
                    sim_time=sim_time, mode_text=mode_text, kp=params.kp, kd=params.kd, 
                    action_scale=params.action_scale, command=current_command, 
                    onnx_input=onnx_input.flatten(), action_raw=action_raw, 
                    final_ctrl=data.ctrl[:]
                )

            # 執行物理步進
            mujoco.mj_step(model, data)

            # 渲染
            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
            cam.lookat = data.body('torso').xpos
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
            mujoco.mjr_render(viewport, scn, con)
            debug_overlay.render(viewport, con)
            glfw.swap_buffers(window)
            glfw.poll_events()
            
        glfw.terminate()

    except Exception as e:
        cprint(f"\nAn error occurred: {e}", "red")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()