import mujoco
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# ==============================================================================
#  使用說明 (v29.1 - 官方文件校驗版)
# ==============================================================================
# 1. 本腳本適用於在 XML 中分離了物理屬性與控制的場景。
# 2. XML 中的 <joint> 標籤負責定義物理屬性 (如 damping, armature)。
#    物理引擎會自動處理它們，Python 代碼無需關心。
# 3. XML 中的 <general gaintype="fixed"> 標籤定義了直接扭矩控制。
# 4. Python 程式碼實現了一個理想的 PD 控制器，其輸出 (torque) 會被施加到
#    這個帶有物理屬性的關節上。
# 5. 調校時，請直接修改下方 CONFIG 中的 KP 和 DAMPING_RATIO。
# ==============================================================================

# --- 全局配置 ---
CONFIG = {
    "MODEL_PATH": "./xmls/pupper_mjx.xml",
    "SIMULATION_DURATION": 2.0,
    "STEP_TARGET_ANGLE": 1.0,
    "JOINT_TO_TEST": "FR_calf_joint", 
    
    # [控制參數] 在此調校你的 PD 控制器性能
    "KP": 10.0,  # 比例增益：決定反應速度和力量，越大反應越快
    "DAMPING_RATIO": 0.3, # 阻尼比：決定系統的穩定性，越大越穩定，越小越容易超調
}

# --- 我們的最終目標 (用於心裡對比) ---
TARGET_METRICS = {
    'overshoot_pct': 0.76, 
    'rise_time': 0.12,
    'peak_time': 0.18, 
    'settling_time': 0.22, 
    'peak_velocity': 7.46,
}

# 加載模型函數不變
def load_model_with_fixed_base(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"錯誤: 模型檔案 '{model_path}' 不存在。")
    with open(model_path, 'r') as f:
        xml_string = f.read()
    xml_string = xml_string.replace('<freejoint/>', '<!-- <freejoint/> removed for testing -->')
    model = mujoco.MjModel.from_xml_string(xml_string)
    return model

# 模擬函數的邏輯被官方文件證實是正確的
def run_manual_pd_simulation(model, data, joint_name, target_pos, duration, kp, damping_ratio):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    actuator_name = joint_name.replace('_joint', '')
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    
    qpos_adr = model.jnt_qposadr[joint_id]
    dof_adr = model.jnt_dofadr[joint_id]

    # 計算 Kd (微分增益)。這個 Kd 是你「控制器」的阻尼項，
    # 它與 XML 中 <joint> 的物理 damping 是獨立的。
    kd = 2 * damping_ratio * math.sqrt(kp)
    print(f"--- 控制器參數 ---")
    print(f"Kp = {kp:.2f}, Damping Ratio = {damping_ratio:.2f} -> 計算出的控制器 Kd = {kd:.2f}")
    
    times, positions, velocities = [], [], []
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0

    while data.time < duration:
        # --- PD 控制器核心邏輯 ---
        current_pos = data.qpos[qpos_adr]
        current_vel = data.qvel[dof_adr]
        
        pos_error = target_pos - current_pos
        
        # 計算控制器應輸出的扭矩
        # 公式: torque = Kp * 位置誤差 - Kd * 當前速度
        pd_torque = kp * pos_error - kd * current_vel
        
        # 將計算出的扭矩賦值給 ctrl。
        # MuJoCo 會將此扭矩施加到由 XML 定義的、具有物理屬性的關節上。
        data.ctrl[actuator_id] = pd_torque
        
        times.append(data.time)
        positions.append(current_pos)
        velocities.append(current_vel)
        
        mujoco.mj_step(model, data)
        
    return np.array(times), np.array(positions), np.array(velocities)

# 分析和繪圖函數不變
def analyze_response(times, positions, velocities, target):
    # ... (代碼省略，與之前完全相同)
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
    # ... (代碼省略，與之前完全相同)
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
    title_str = (f'Manual PD Tuning - Joint: {joint_name} (Kp={CONFIG["KP"]}, D_ratio={CONFIG["DAMPING_RATIO"]})')
    plt.title(title_str, fontsize=16)
    fig.tight_layout()
    plt.show()


def main():
    print(f"🔧 開始手動 PD 控制器調校測試...")
    print(f"   - 正在載入模型: {CONFIG['MODEL_PATH']}")
    print(f"   - 測試目標關節: {CONFIG['JOINT_TO_TEST']}")

    try:
        model = load_model_with_fixed_base(CONFIG["MODEL_PATH"])
        data = mujoco.MjData(model)
        
        # 打印出模型中的物理阻尼，以供參考
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, CONFIG["JOINT_TO_TEST"])
        print("--- 從 XML 讀取的物理屬性 ---")
        print(f"關節物理阻尼 (damping): {model.jnt_damping[joint_id]:.4f}")
        print(f"關節電樞慣量 (armature): {model.dof_armature[model.jnt_dofadr[joint_id]]:.4f}")
        print(f"✅ 模型載入成功！")
        
    except Exception as e:
        print(f"❌ 錯誤: 無法載入或處理模型檔案 '{CONFIG['MODEL_PATH']}'.")
        print(f"   詳細錯誤: {e}")
        return

    joint_name = CONFIG["JOINT_TO_TEST"]

    print("\n🚀 正在執行模擬 (使用 Python 實現的 PD 控制)...")
    times, positions, velocities = run_manual_pd_simulation(
        model, data, joint_name, 
        CONFIG["STEP_TARGET_ANGLE"], 
        CONFIG["SIMULATION_DURATION"],
        CONFIG["KP"],
        CONFIG["DAMPING_RATIO"]
    )
    
    print("📊 正在分析結果...")
    metrics = analyze_response(times, positions, velocities, CONFIG["STEP_TARGET_ANGLE"])
    
    print("\n--- 控制器性能報告 ---")
    print(f"上升時間 (10%-90%): {metrics.get('rise_time', 'N/A'):.4f} s  (目標: ~{TARGET_METRICS['rise_time']:.2f}s)")
    print(f"峰值時間:           {metrics.get('peak_time', 'N/A'):.4f} s  (目標: ~{TARGET_METRICS['peak_time']:.2f}s)")
    print(f"超調量:             {metrics.get('overshoot_pct', 'N/A'):.2f} %   (目標: ~{TARGET_METRICS['overshoot_pct']:.2f}%)")
    print(f"整定時間 (±2%):     {metrics.get('settling_time', 'N/A'):.4f} s  (目標: ~{TARGET_METRICS['settling_time']:.2f}s)")
    print(f"峰值速度:           {metrics.get('peak_velocity', 'N/A'):.4f} rad/s (目標: ~{TARGET_METRICS['peak_velocity']:.2f} rad/s)")
    print("------------------------\n")
    
    print("📈 正在繪製最終響應圖...")
    plot_response(times, positions, velocities, metrics, joint_name, CONFIG["STEP_TARGET_ANGLE"])
    
    print("\n✅ 測試完成。請查看彈出的圖表。")

if __name__ == "__main__":
    main()