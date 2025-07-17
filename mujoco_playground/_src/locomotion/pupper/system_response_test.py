import mujoco
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import time

# ==============================================================================
#  使用說明 (v28.2 - 最終手動微調版，修正 API 兼容性)
# ==============================================================================
# 1. 此腳本用於最後的專家微調階段。
# 2. 直接手動修改您的 'pupper_mjx.xml' 檔案中的 kp 和 dampratio 值。
# 3. 運行此腳本，觀察生成的圖表和性能指標，以判斷您的修改是否達標。
# ==============================================================================

# --- 全局配置 ---
CONFIG = {
    "MODEL_PATH": "./xmls/pupper_mjx.xml",
    "SIMULATION_DURATION": 2.0,
    "STEP_TARGET_ANGLE": 1.0,
    "JOINT_TO_TEST": "FR_calf_joint", 
}

# --- 我們的最終目標 (用於心裡對比) ---
TARGET_METRICS = {
    'overshoot_pct': 0.76, 
    'rise_time': 0.12,
    'peak_time': 0.18, 
    'settling_time': 0.22, 
    'peak_velocity': 7.46,
}

def load_model_with_fixed_base(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"錯誤: 模型檔案 '{model_path}' 不存在。")
    with open(model_path, 'r') as f:
        xml_string = f.read()
    
    xml_string = xml_string.replace('<freejoint/>', '<!-- <freejoint/> removed for testing -->')
    
    model = mujoco.MjModel.from_xml_string(xml_string)
    return model

# 模擬、分析函數與之前完全相同
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
        target_joint_id = model.actuator_trnid[i, 0]
        joint_to_actuator_map[target_joint_id] = i

    times, positions, velocities = [], [], []
    mujoco.mj_resetData(model, data)
    while data.time < duration:
        for jid in movable_joint_ids:
            if jid != joint_id:
                act_id_to_lock = joint_to_actuator_map.get(jid)
                if act_id_to_lock is not None and act_id_to_lock < model.nu:
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

# [MODIFIED] 簡化繪圖函數，不再從模型中讀取易出錯的 API 屬性
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
    title_str = (f'Manual Tuning Result - Joint: {joint_name}') # 簡化標題
    plt.title(title_str, fontsize=16)
    fig.tight_layout()
    plt.show()

def main():
    print(f"🔧 開始手動調校測試...")
    print(f"   - 正在載入模型: {CONFIG['MODEL_PATH']}")
    print(f"   - 測試目標關節: {CONFIG['JOINT_TO_TEST']}")

    try:
        model = load_model_with_fixed_base(CONFIG["MODEL_PATH"])
        data = mujoco.MjData(model)
        print(f"✅ 模型載入成功！")
    except Exception as e:
        print(f"❌ 錯誤: 無法載入或處理模型檔案 '{CONFIG['MODEL_PATH']}'.")
        print(f"   詳細錯誤: {e}")
        return

    joint_name = CONFIG["JOINT_TO_TEST"]

    print("\n🚀 正在執行模擬...")
    times, positions, velocities = run_step_response_simulation(
        model, data, joint_name, CONFIG["STEP_TARGET_ANGLE"], CONFIG["SIMULATION_DURATION"]
    )
    
    print("📊 正在分析結果...")
    metrics = analyze_response(times, positions, velocities, CONFIG["STEP_TARGET_ANGLE"])
    
    print("\n--- 手動調校結果報告 ---")
    print(f"上升時間 (10%-90%): {metrics.get('rise_time', 'N/A'):.4f} s  (目標: ~{TARGET_METRICS['rise_time']:.2f}s)")
    print(f"峰值時間:           {metrics.get('peak_time', 'N/A'):.4f} s  (目標: ~{TARGET_METRICS['peak_time']:.2f}s)")
    print(f"超調量:             {metrics.get('overshoot_pct', 'N/A'):.2f} %   (目標: ~{TARGET_METRICS['overshoot_pct']:.2f}%)")
    print(f"整定時間 (±2%):     {metrics.get('settling_time', 'N/A'):.4f} s  (目標: ~{TARGET_METRICS['settling_time']:.2f}s)")
    print(f"峰值速度:           {metrics.get('peak_velocity', 'N/A'):.4f} rad/s (目標: ~{TARGET_METRICS['peak_velocity']:.2f} rad/s)")
    print("------------------------\n")
    
    print("📈 正在繪製最終響應圖...")
    # [MODIFIED] 簡化函數呼叫
    plot_response(times, positions, velocities, metrics, joint_name, CONFIG["STEP_TARGET_ANGLE"])
    
    print("\n✅ 測試完成。請查看彈出的圖表。")

if __name__ == "__main__":
    main()