import mujoco
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import optuna
import time

# ==============================================================================
#  使用說明 (v35 - 終極版：帶低通濾波的手動 PD 扭矩控制)
# ==============================================================================
# 1. 前提：使用基於 <general> 驅動器的 XML，<option> 標籤保持不變。
# 2. 此腳本在 Python 中實現帶低通濾波的 PD 控制，以在您給定的物理引擎
#    設定下，尋找穩定的最佳解。
# 3. 優化目標變為三個參數：kp, kd, 和濾波時間常數 tau。
# ==============================================================================

# --- 全局配置 ---
CONFIG = {
    "MODEL_PATH": "./xmls/pupper_mjx.xml",
    "SIMULATION_DURATION": 2.0,
    "STEP_TARGET_ANGLE": 1.0,
    "JOINT_TO_OPTIMIZE": "FR_calf_joint",
    "N_TRIALS": 300, # 三個參數需要更多的嘗試
}

# --- 優化目標 ---
TARGET_METRICS = {
    'overshoot_pct': 0.76, 'rise_time': 0.12,
    'settling_time': 0.22, 'peak_velocity': 7.46,
}

# --- 成本函數權重 ---
COST_WEIGHTS = {
    'overshoot_pct': 3.0, 'rise_time': 5.0,
    'settling_time': 4.0, 'peak_velocity': 1.0,
}

# --- 硬性約束 ---
HARD_CONSTRAINTS = { 'max_overshoot_pct': 5.0, }

def load_base_model(model_path):
    """只負責載入和準備基礎模型。"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"錯誤: 模型檔案 '{model_path}' 不存在。")
    with open(model_path, 'r') as f:
        xml_string = f.read()
    xml_string = xml_string.replace('<freejoint/>', '<!-- <freejoint/> removed -->')
    model = mujoco.MjModel.from_xml_string(xml_string)
    return model

# [MODIFIED] 模擬函數現在實現帶低通濾波的 PD 控制
def run_step_response_simulation(model, data, joint_name, kp, kd, tau, target_pos, duration):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    actuator_name = joint_name.replace('_joint', '_torque')
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    
    joint_to_actuator_map = {
        model.actuator_trnid[i, 0]: i
        for i in range(model.nu)
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
    }
    
    mujoco.mj_resetData(model, data)
    
    # 低通濾波器狀態
    filtered_torque = 0.0
    dt = model.opt.timestep

    times, positions, velocities = [], [], []
    while data.time < duration:
        for jid, act_id in joint_to_actuator_map.items():
            if jid != joint_id:
                data.ctrl[act_id] = 0.0
        
        current_pos = data.qpos[model.jnt_qposadr[joint_id]]
        current_vel = data.qvel[model.jnt_dofadr[joint_id]]
        pos_error = target_pos - current_pos
        
        # === 帶低通濾波的 PD 控制核心邏輯 ===
        # 1. 計算原始的、未經濾波的 PD 力矩
        raw_torque = kp * pos_error - kd * current_vel
        
        # 2. 應用一階低通濾波器
        # alpha 決定了濾波的平滑程度
        alpha = dt / (tau + dt)
        filtered_torque = alpha * raw_torque + (1 - alpha) * filtered_torque
        
        # 3. 將平滑後的力矩施加到致動器
        torque_to_apply = filtered_torque
        # ====================================
        
        data.ctrl[actuator_id] = torque_to_apply

        times.append(data.time)
        positions.append(current_pos)
        velocities.append(current_vel)
        
        mujoco.mj_step(model, data)
        
        # 檢查模擬是否穩定
        if not np.isfinite(data.qacc).all():
             # 如果不穩定，提前終止並返回目前為止的數據
             print(f"警告: 偵測到不穩定 (QACC 非有限值)，在 time={data.time:.4f} 提前終止。")
             return np.array(times), np.array(positions), np.array(velocities)

    return np.array(times), np.array(positions), np.array(velocities)

# ... (analyze_response 和 plot_response 函數保持不變) ...
def analyze_response(times, positions, velocities, target):
    metrics = {}
    if not isinstance(times, np.ndarray) or times.size == 0: return {}
    if not isinstance(positions, np.ndarray) or positions.size == 0: return {}
    if not isinstance(velocities, np.ndarray) or velocities.size == 0: return {}
    
    try:
        t10_indices = np.where(positions >= 0.1 * target)[0]
        t90_indices = np.where(positions >= 0.9 * target)[0]
        if t10_indices.size > 0 and t90_indices.size > 0:
            metrics['rise_time'] = times[t90_indices[0]] - times[t10_indices[0]]
        else:
            metrics['rise_time'] = float('inf')
    except IndexError:
        metrics['rise_time'] = float('inf')
    peak_idx = np.argmax(positions)
    metrics['peak_time'] = times[peak_idx]
    peak_value = positions[peak_idx]
    metrics['overshoot_pct'] = max(0, (peak_value - target) / target * 100)
    settling_band_upper = target * 1.02
    settling_band_lower = target * 0.98
    after_peak_times_indices = np.where(times >= metrics['peak_time'])[0]
    if after_peak_times_indices.size > 0:
        first_after_peak_idx = after_peak_times_indices[0]
        outside_band_indices = np.where(
            (positions[first_after_peak_idx:] > settling_band_upper) |
            (positions[first_after_peak_idx:] < settling_band_lower)
        )[0]
        if outside_band_indices.size > 0:
            last_outside_global_idx = first_after_peak_idx + outside_band_indices[-1]
            metrics['settling_time'] = times[last_outside_global_idx]
        else:
            metrics['settling_time'] = metrics['peak_time']
    else:
        metrics['settling_time'] = float('inf')
    metrics['peak_velocity'] = np.max(np.abs(velocities))
    return metrics

def plot_response(times, positions, velocities, metrics, joint_name, target, model_for_report, best_kp, best_kd, best_tau):
    joint_id = mujoco.mj_name2id(model_for_report, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    dof_id = model_for_report.jnt_dofadr[joint_id]
    fixed_armature = model_for_report.dof_armature[dof_id]
    fixed_damping = model_for_report.dof_damping[dof_id]
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
    title_str = (f'Optimized Response (Filtered PD Torque Control)\n'
                 f'Joint: {joint_name} | Kp={best_kp:.2f}, Kd={best_kd:.2f}, Tau={best_tau:.4f}')
    plt.title(title_str, fontsize=16)
    fig.tight_layout()
    plt.show()

def optimize_joint_parameters(model, joint_name_to_optimize, n_trials):
    print(f"\n{'='*60}")
    print(f"🚀 [帶濾波的PD扭矩控制模式] 開始為 '{joint_name_to_optimize}' 尋找最佳 Kp/Kd/Tau ({n_trials} 次嘗試)...")
    start_time = time.time()

    def objective(trial):
        # 優化三個參數
        kp = trial.suggest_float("kp", 1.0, 300.0) 
        kd = trial.suggest_float("kd", 0.1, 10.0)
        # tau 是濾波器的時間常數，tau 越大，輸出越平滑
        tau = trial.suggest_float("tau", 0.001, 0.05, log=True)

        try:
            data = mujoco.MjData(model)
            times, positions, velocities = run_step_response_simulation(
                model, data, joint_name_to_optimize, kp, kd, tau, CONFIG["STEP_TARGET_ANGLE"], CONFIG["SIMULATION_DURATION"]
            )
        except Exception as e:
            print(f"模擬失敗，參數: kp={kp:.2f}, kd={kd:.2f}, tau={tau:.4f}。錯誤: {e}")
            return float('inf')

        sim_metrics = analyze_response(times, positions, velocities, CONFIG["STEP_TARGET_ANGLE"])
        if not sim_metrics: return float('inf')
        
        # 如果模擬提前終止（不穩定），給予巨大懲罰
        if times[-1] < (CONFIG["SIMULATION_DURATION"] - model.opt.timestep * 2):
            return 1e7

        if sim_metrics['overshoot_pct'] > HARD_CONSTRAINTS['max_overshoot_pct']:
            return 1e6 + (sim_metrics['overshoot_pct'] * 10)

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
    print(f"  - 找到的最佳參數: kp={best_params['kp']:.2f}, kd={best_params['kd']:.2f}, tau={best_params['tau']:.4f}")
    print(f"  - 最低成本值: {study.best_trial.value:.4f}")
    
    return best_params

def main():
    try:
        model = load_base_model(CONFIG["MODEL_PATH"])
        print(f"✅ 成功載入並處理模型: {CONFIG['MODEL_PATH']}")
    except Exception as e:
        print(f"❌ 錯誤: 無法載入或處理模型檔案 '{CONFIG['MODEL_PATH']}'.")
        print(f"   詳細錯誤: {e}")
        return

    joint_name = CONFIG["JOINT_TO_OPTIMIZE"]
    best_params = optimize_joint_parameters(
        model,
        joint_name, 
        CONFIG["N_TRIALS"]
    )

    print(f"\n{'='*60}")
    print(f"🎉🎉🎉 [帶濾波的PD扭矩控制模式] 參數優化已全部完成！🎉🎉🎉")
    print(f"{'='*60}")
    print("最終找到的最佳控制器參數總結：")
    
    kp_val, kd_val, tau_val = best_params['kp'], best_params['kd'], best_params['tau']
    print(f"\n關節: {joint_name}")
    print(f"  - kp:  {kp_val:.4f} (比例增益)")
    print(f"  - kd:  {kd_val:.4f} (微分增益)")
    print(f"  - tau: {tau_val:.4f} (力矩濾波時間常數)")
    
    print("\n這組 (kp, kd, tau) 參數可以直接在您的 Python/JAX 控制器中使用。")
    print(f"{'='*60}")

    print("\n📈 正在為該關節繪製其最佳響應圖...")
    data = mujoco.MjData(model)
    times, positions, velocities = run_step_response_simulation(
        model, data, joint_name, kp_val, kd_val, tau_val, CONFIG["STEP_TARGET_ANGLE"], CONFIG["SIMULATION_DURATION"]
    )
    metrics = analyze_response(times, positions, velocities, CONFIG["STEP_TARGET_ANGLE"])
    plot_response(times, positions, velocities, metrics, joint_name, CONFIG["STEP_TARGET_ANGLE"], model, kp_val, kd_val, tau_val)

if __name__ == "__main__":
    main()