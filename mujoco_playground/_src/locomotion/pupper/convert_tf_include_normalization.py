# ===================================================================
#      JAX to ONNX Conversion (最終端到端版 - 包含正規化和歷史處理)
# ===================================================================

# --- 導入 (保持不變) ---
import pickle
import numpy as np
import os
from etils import epath
from typing import List, Tuple

# 強制 JAX 和 TensorFlow 使用 CPU
print("--- Forcing JAX and TensorFlow to use CPU to ensure compatibility ---")
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import tensorflow as tf
from tensorflow.keras import layers
import tf2onnx
import onnx

# --- 輔助類和函數 ---
class KerasMLP(tf.keras.Model):
    """一個與 Brax 的 MLP 行為一致的 Keras 模型。"""
    def __init__(self, layer_sizes: List[int], activation=tf.nn.swish, name='mlp_block'):
        super().__init__(name=name)
        self.mlp_layers = []
        for i, size in enumerate(layer_sizes):
            act = activation if i < len(layer_sizes) - 1 else None
            self.mlp_layers.append(layers.Dense(
                size, activation=act, kernel_initializer='lecun_uniform', name=f"hidden_{i}"
            ))
    def call(self, inputs):
        x = inputs
        for layer in self.mlp_layers:
            x = layer(x)
        return x

def make_end_to_end_tf_policy(
    total_obs_size: int, 
    single_step_obs_size: int,
    act_size: int, 
    hidden_sizes: Tuple[int, ...], 
    normalizer_mean_tiled: np.ndarray, 
    normalizer_std_tiled: np.ndarray
):
    """
    創建一個完整的、端到端的 TF 策略網路。
    它接收 720 維的原始歷史觀測，並在內部完成所有處理。
    """
    # 1. 輸入層接收 720 維的扁平化原始觀測
    inputs = {'state': tf.keras.Input(shape=(total_obs_size,), name='state')}
    
    # 2. 正規化層，使用 720 維的 mean 和 std
    normalized_obs_720d = layers.Lambda(
        lambda x: (x['state'] - normalizer_mean_tiled) / (normalizer_std_tiled + 1e-8),
        name="normalization"
    )(inputs)
    
    # 3. 【關鍵】提取層：從 720 維的歸一化觀測中，只取出最後 48 維
    #    這對應於最新的、被正確歸一化的觀測
    latest_normalized_obs_48d = layers.Lambda(
        lambda x: x[:, -single_step_obs_size:],
        name="extract_latest_obs"
    )(normalized_obs_720d)
    
    # 4. 核心 MLP，接收 48 維的數據
    mlp = KerasMLP(layer_sizes=list(hidden_sizes) + [act_size * 2])
    logits = mlp(latest_normalized_obs_48d)
    
    # 5. 輸出層
    loc = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1)[0])(logits)
    outputs = tf.keras.layers.Activation('tanh', name='action')(loc)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="PupperPPOEndToEndPolicy")

def transfer_weights(jax_policy_params, tf_model):
    # 這個函數現在的目標是 tf_model 中的 'mlp_block'
    # 它的邏輯與我們之前的簡單版本完全一樣
    tf_mlp_layers = [l for l in tf_model.get_layer('mlp_block').layers if isinstance(l, layers.Dense)]
    jax_weights_flat = jax.tree_util.tree_leaves(jax_policy_params)

    if len(tf_mlp_layers) * 2 != len(jax_weights_flat):
        print(f"  [致命錯誤] TF 層數 ({len(tf_mlp_layers)}) 與 JAX 權重數 ({len(jax_weights_flat)//2}) 不匹配!")
        return False
    
    for i, tf_layer in enumerate(tf_mlp_layers):
        jax_bias = np.array(jax_weights_flat[i * 2])
        jax_kernel = np.array(jax_weights_flat[i * 2 + 1])
        if jax_kernel.shape != tf_layer.get_weights()[0].shape:
            return False # 簡化錯誤處理
        tf_layer.set_weights([jax_kernel, jax_bias])
    return True

def main():
    """主函數"""
    # --- 步驟 0: 配置 ---
    print("\n--- Step 0: Configuration ---")
    env_name = 'PupperJoystickWithGun'
    CHECKPOINT_DIR = epath.Path("checkpoints/" + env_name).resolve()
    
    SINGLE_STEP_OBS_SIZE = 51
    HISTORY_LEN = 15
    POLICY_OBS_SIZE = SINGLE_STEP_OBS_SIZE * HISTORY_LEN # 720 ->750 -> 765
    ACTION_SIZE = 12
    POLICY_HIDDEN_LAYER_SIZES = (512, 256, 128)

    # --- 步驟 1: 載入 JAX 參數 ---
    print(f"\n--- Step 1: Loading JAX parameters ---")
    # ... (載入參數的邏輯保持不變) ...
    try:
        steps = [int(p.name) for p in CHECKPOINT_DIR.iterdir() if p.is_dir() and p.name.isdigit()]
        latest_step = 101580800 #max(steps) if steps else None
        if not latest_step: raise FileNotFoundError("找不到任何 checkpoint。")

        pkl_path = CHECKPOINT_DIR / str(latest_step) / "params.pkl"
        print(f"  - 從最新的 checkpoint 載入: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            params_tuple = pickle.load(f)
        normalizer_params, policy_params, _ = params_tuple
        print("  - JAX 參數載入成功。")
    except Exception as e:
        print(f"  [致命錯誤] 無法載入參數: {e}")
        raise

    # --- 步驟 2: 構建端到端的 TF 模型 ---
    print("\n--- Step 2: Building END-TO-END TF model ---")
    try:
        # 1. 提取 48 維的 mean 和 std
        original_mean = np.array(normalizer_params.mean['state'])
        original_std = np.array(normalizer_params.std['state'])
        
        # 2. 手動將 Normalizer 參數擴展到 720 ->750 維
        tiled_mean = np.tile(original_mean, HISTORY_LEN)
        tiled_std = np.tile(original_std, HISTORY_LEN)
        
        # 3. 構建包含所有邏輯的端到端模型
        tf_policy_network = make_end_to_end_tf_policy(
            total_obs_size=POLICY_OBS_SIZE,
            single_step_obs_size=SINGLE_STEP_OBS_SIZE,
            act_size=ACTION_SIZE,
            hidden_sizes=POLICY_HIDDEN_LAYER_SIZES,
            normalizer_mean_tiled=tiled_mean,
            normalizer_std_tiled=tiled_std
        )
        print("  - 端到端 TensorFlow Keras 模型構建成功。")
        tf_policy_network.summary()
    except Exception as e:
        print(f"  [錯誤] 構建 TF 模型失敗: {e}")
        raise

    # --- 步驟 3: 轉移權重到核心 MLP 部分 ---
    print("\n--- Step 3: Transferring JAX weights to the CORE MLP of the model ---")
    if not transfer_weights(policy_params, tf_policy_network):
        return
    print("  - 權重轉移完成。")

    # --- 步驟 4: 將完整的 TF 模型轉換為 ONNX ---
    print("\n--- Step 4: Converting the complete TensorFlow model to ONNX ---")
    output_dir = epath.Path("onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pupper_ppo_policy_e2e_{latest_step}.onnx"
    spec = ({'state': tf.TensorSpec((None, POLICY_OBS_SIZE), tf.float32, name="state")},)
    try:
        model_proto, _ = tf2onnx.convert.from_keras(
            tf_policy_network, 
            input_signature=spec, 
            opset=13,
            output_path=str(output_path)
        )
        print(f"\n  - 轉換成功! 端到端 ONNX 模型已保存至: {output_path}")
        onnx.checker.check_model(str(output_path))
        print("  - ONNX 模型檢查成功。")
    except Exception as e:
        print(f"  [錯誤] ONNX 轉換過程中發生錯誤: {e}")
        raise

if __name__ == "__main__":
    main()