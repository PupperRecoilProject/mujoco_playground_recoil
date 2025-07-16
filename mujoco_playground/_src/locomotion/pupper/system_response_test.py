import mujoco
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import optuna
import time

# ==============================================================================
#  使用說明 (v13 - 專注優化 Lower Leg)
# ==============================================================================
# 1. 確保已安裝 optuna: `pip install optuna`
# 2. 此腳本將集中所有計算資源，只為 "FR_calf_joint" 尋找最佳參數。
# 3. 整個過程無中斷，結束後會生成最終報告和可複製的 XML 片段。
# ==============================================================================

# --- 全局配置 ---
CONFIG = {
    "MODEL_PATH": "./xmls/pupper_mjx.xml",
    "SIMULATION_DURATION": 2.0,
    "STEP_TARGET_ANGLE": 1.0,
    # [ACTION] 專注於我們擁有真實數據的這一個關節
    "JOINTS_TO_OPTIMIZE": [
        "FR_calf_joint",
    ],
    # [ACTION] 增加嘗試次數，以進行更精細的搜索
    "N_TRIALS_PER_JOINT": 1000, 
}

# --- 優化目標 (來自您的真實世界數據 for lower leg) ---
TARGET_METRICS = {
    'overshoot_pct': 0.76,
    'rise_time': 0.12,
    'settling_time': 0.16,
    'peak_velocity': 7.46,
}

# --- 成本函數權重 ---
COST_WEIGHTS = {
    'overshoot_pct': 2.0, 'rise_time': 3.0,
    'settling_time': 1.0, 'peak_velocity': 0.5,
}

# [MODIFIED] 函數新增了 armature 參數
def load_model_with_fixed_base(model_path, joint_name_to_tune, kp, damping, armature):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"錯誤: 模型檔案 '{model_path}' 不存在。")
    with open(model_path, 'r') as f:
        xml_string = f.read()
    xml_string = xml_string.replace('meshdir="assets"', 'meshdir="xmls/assets"')
    xml_string = xml_string.replace('<freejoint/>', '<!-- <freejoint/> removed -->')
    
    joint_class_map = {'hip': 'abduction', 'thigh': 'hip', 'calf': 'knee'}
    class_to_tune = None
    for keyword, classname in joint_class_map.items():
        if keyword in joint_name_to_tune:
            class_to_tune = classname
            break
    if not class_to_tune:
        raise ValueError(f"無法為關節 '{joint_name_to_tune}' 找到對應的 class")

    # 動態設定 Kp, Damping, 和 Armature
    # 注意: 我們現在只會修改 "knee" 這個 class
    xml_string = re.sub(rf'(<default class="{class_to_tune}">.*?<joint.*?damping=")[^"]*(".*)', rf'\g<1>{damping}\g<2>', xml_string, flags=re.S)
    xml_string = re.sub(rf'(<default class="{class_to_tune}">.*?<position.*?kp=")[^"]*(".*)', rf'\g<1>{kp}\g<2>', xml_string, flags=re.S)
    xml_string = re.sub(rf'(<default class="{class_to_tune}">.*?<joint.*?armature=")[^"]*(".*)', rf'\g<1>{armature}\g<2>', xml_string, flags=re.S)
    
    model = mujoco.MjModel.from_xml_string(xml_string)
    return model

# ... 其他輔助函數保持不變 ...
def run_step_response_simulation(model, data, joint_name, target_pos, duration):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    actuator_name = joint_name.replace('_joint', '')
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    initial_qpos = data.qpos.copy()
    dof_adrs = model.jnt_dofadr
    dof_adrs_ext = np.append(dof_adrs, model.nv)
    dof_nums = dof_adrs_ext[1:] - dof_adrs_ext[:-1]
    movable_joint_ids = [i for i, num in enumerate(dof_nums) if num > 0]
    joint_to_actuator_map = {}
    for i in range(model.nu):
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            joint_to_actuator_map[model.actuator_trnid[i, 0]] = i
    times, positions, velocities = [], [], []
    while data.time < duration:
        for jid in movable_joint_ids:
            if jid != joint_id:
                act_id_to_lock = joint_to_actuator_map.get(jid, -1)
                if act_id_to_lock > -1:
                    data.ctrl[act_id_to_lock] = initial_qpos[model.jnt_qposadr[jid]]
        data.ctrl[actuator_id] = target_pos
        times.append(data.time)
        positions.append(data.qpos[model.jnt_qposadr[joint_id]])
        velocities.append(data.qvel[model.jnt_dofadr[joint_id]])
        mujoco.mj_step(model, data)
    return np.array(times), np.array(positions), np.array(velocities)

def analyze_response(times, positions, velocities, target):
    metrics = {}
    if target == 0: return metrics
    try:
        t10_idx = np.where(positions >= 0.1 * target)[0][0]
        t90_idx = np.where(positions >= 0.9 * target)[0][0]
        metrics['rise_time'] = times[t90_idx] - times[t10_idx]
    except IndexError: metrics['rise_time'] = float('inf')
    peak_idx = np.argmax(positions)
    metrics['peak_time'] = times[peak_idx]
    peak_value = positions[peak_idx]
    metrics['overshoot_pct'] = max(0, (peak_value - target) / target * 100)
    settling_band_upper = target * 1.02
    settling_band_lower = target * 0.98
    after_peak_indices = np.where(times > metrics['peak_time'])[0]
    if len(after_peak_indices) > 0:
      search_space = positions[after_peak_indices[0]:]
      outside_after_peak = np.where((search_space > settling_band_upper) | (search_space < settling_band_lower))[0]
      if len(outside_after_peak) > 0:
        last_outside_idx = after_peak_indices[0] + outside_after_peak[-1]
        metrics['settling_time'] = times[last_outside_idx]
      else: 
        try:
           settled_idx = np.where(positions >= 0.98 * target)[0][0]
           metrics['settling_time'] = times[settled_idx]
        except IndexError: metrics['settling_time'] = float('inf')
    else: metrics['settling_time'] = float('inf')
    metrics['peak_velocity'] = np.max(np.abs(velocities))
    return metrics

def plot_response(times, positions, velocities, metrics, joint_name, target):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(times, positions, '.-', color='royalblue', label='Actual Position')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Position (rad)', color='royalblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.grid(True, which='both', linestyle=':')
    ax1.axhline(y=target, color='darkgreen', linestyle='--', label=f'Target Position ({target:.1f} rad)')
    ax1.axhspan(target * 0.98, target * 1.02, color='lightgreen', alpha=0.5, label='Settling Band (±2%)')
    if metrics.get('settling_time') and not np.isnan(metrics['settling_time']) and not np.isinf(metrics['settling_time']):
        ax1.axvline(x=metrics['settling_time'], color='gold', linestyle='--', label=f'Settling Time: {metrics["settling_time"]:.2f}s')
    ax2 = ax1.twinx()
    ax2.plot(times, velocities, '.--', color='crimson', label='Actual Velocity')
    ax2.set_ylabel('Velocity (rad/s)', color='crimson', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='crimson')
    if metrics.get('peak_time'):
        peak_time = metrics['peak_time']
        peak_val = target * (1 + metrics['overshoot_pct'] / 100)
        ax1.plot(peak_time, peak_val, 'o', color='red')
        annotation_text = (f"Peak: {peak_val:.3f} rad\n"
                           f"Time: {peak_time:.2f} s\n"
                           f"Overshoot: {metrics['overshoot_pct']:.2f}%")
        ax1.annotate(annotation_text, xy=(peak_time, peak_val),
                     xytext=(peak_time + 0.15 * max(times), peak_val * 0.8),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.9),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='black'))
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower right')
    plt.title(f'Optimized Step Response - Joint: {joint_name}', fontsize=16)
    fig.tight_layout()
    plt.show()

# ==============================================================================
#                      新的、模組化的優化函數
# ==============================================================================
def optimize_joint_parameters(joint_name_to_optimize, n_trials):
    print(f"\n{'='*60}")
    print(f"🚀 開始為關節 '{joint_name_to_optimize}' 自動尋找最佳參數 ({n_trials} 次嘗試)...")
    start_time = time.time()

    def objective(trial):
        kp = trial.suggest_float("kp", 50, 800)
        damping = trial.suggest_float("damping", 0.1, 5.0)
        armature = trial.suggest_float("armature", 0.001, 0.1)

        try:
            model = load_model_with_fixed_base(
                CONFIG["MODEL_PATH"], joint_name_to_optimize, kp, damping, armature
            )
            data = mujoco.MjData(model)
            times, positions, velocities = run_step_response_simulation(
                model, data, joint_name_to_optimize, CONFIG["STEP_TARGET_ANGLE"], CONFIG["SIMULATION_DURATION"]
            )
        except Exception:
            return float('inf')

        sim_metrics = analyze_response(times, positions, velocities, CONFIG["STEP_TARGET_ANGLE"])
        if not sim_metrics: return float('inf')

        cost = 0
        for key, target_val in TARGET_METRICS.items():
            sim_val = sim_metrics.get(key, float('inf'))
            weight = COST_WEIGHTS.get(key, 1.0)
            error = ((sim_val - target_val) / (target_val + 1e-6))**2
            cost += weight * error
        
        return cost

    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    end_time = time.time()
    print(f"✅ 優化完成！耗時 {end_time - start_time:.2f} 秒。")

    best_params = study.best_trial.params
    print(f"  - 找到的最佳參數: kp={best_params['kp']:.2f}, damping={best_params['damping']:.2f}, armature={best_params['armature']:.4f}")
    print(f"  - 最低成本值: {study.best_trial.value:.4f}")
    
    return best_params

# ==============================================================================
#                           新的「總指揮官」主函數
# ==============================================================================
def main():
    all_best_params = {}

    # 只對我們指定的單一關節進行優化
    joint_name = CONFIG["JOINTS_TO_OPTIMIZE"][0]
    best_params = optimize_joint_parameters(
        joint_name, 
        CONFIG["N_TRIALS_PER_JOINT"]
    )
    all_best_params[joint_name] = best_params

    print(f"\n{'='*60}")
    print("🎉🎉🎉 參數優化已全部完成！🎉🎉🎉")
    print(f"{'='*60}")
    print("最終找到的最佳參數總結：")
    
    joint_class_map = {'hip': 'abduction', 'thigh': 'hip', 'calf': 'knee'}
    
    # 只打印和繪製我們優化的那個關節
    params = all_best_params[joint_name]
    kp_val, d_val, arm_val = params['kp'], params['damping'], params['armature']
    print(f"\n關節: {joint_name}")
    print(f"  - kp:       {kp_val:.4f}")
    print(f"  - damping:  {d_val:.4f}")
    print(f"  - armature: {arm_val:.4f}")

    class_to_tune = None
    for keyword, classname in joint_class_map.items():
        if keyword in joint_name:
            class_to_tune = classname
            break
    
    if class_to_tune:
        snippet = f'''
  <!-- {joint_name} - Optimized Parameters -->
  <default class="{class_to_tune}">
    <joint damping="{d_val:.4f}" armature="{arm_val:.4f}"/> 
    <position kp="{kp_val:.4f}"/> 
  </default>'''
        print(f"\n{'='*60}")
        print("您可以將以下程式碼片段直接複製到您的 XML 檔案的 <default> 區塊中，\n以更新 'knee' class 的設定：")
        print(f"--- 複製開始 ---")
        print(snippet)
        print(f"--- 複製結束 ---")
        print(f"{'='*60}")
    
    print("\n📈 正在為該關節繪製其最佳響應圖...")
    model = load_model_with_fixed_base(CONFIG["MODEL_PATH"], joint_name, params['kp'], params['damping'], params['armature'])
    data = mujoco.MjData(model)
    times, positions, velocities = run_step_response_simulation(
        model, data, joint_name, CONFIG["STEP_TARGET_ANGLE"], CONFIG["SIMULATION_DURATION"]
    )
    metrics = analyze_response(times, positions, velocities, CONFIG["STEP_TARGET_ANGLE"])
    plot_response(times, positions, velocities, metrics, joint_name, CONFIG["STEP_TARGET_ANGLE"])


if __name__ == "__main__":
    main()