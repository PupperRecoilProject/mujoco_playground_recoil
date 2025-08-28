# ===================================================================
#      JAX to ONNX Conversion (最終版 - 導出核心 MLP 和獨立 Normalizer)
# ===================================================================

# --- 導入 ---
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

def make_tf_core_policy_network(policy_obs_size: int, act_size: int, hidden_sizes: Tuple[int, ...]):
    """創建一個【不包含】Normalizer 的核心策略網路。"""
    # 輸入層接收【已經被歸一化】的數據
    inputs = {'state': tf.keras.Input(shape=(policy_obs_size,), name='state')}
    
    # 直接將輸入送入 MLP
    mlp = KerasMLP(layer_sizes=list(hidden_sizes) + [act_size * 2])
    logits = mlp(inputs['state']) # 直接傳遞張量
    
    loc = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1)[0])(logits)
    outputs = tf.keras.layers.Activation('tanh', name='action')(loc)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="PupperPPOCorePolicy")

def transfer_weights(jax_policy_params, tf_model):
    """將 JAX 權重轉移到 TensorFlow 模型。"""
    tf_mlp_layers = [l for l in tf_model.get_layer('mlp_block').layers if isinstance(l, layers.Dense)]
    jax_weights_flat = jax.tree_util.tree_leaves(jax_policy_params)

    if len(tf_mlp_layers) * 2 != len(jax_weights_flat):
        print(f"  [致命錯誤] TF 層數 ({len(tf_mlp_layers)}) 與 JAX 權重數 ({len(jax_weights_flat)//2}) 不匹配!")
        return False
    
    for i, tf_layer in enumerate(tf_mlp_layers):
        # Brax 權重順序是 (bias, kernel)
        jax_bias = np.array(jax_weights_flat[i * 2])
        jax_kernel = np.array(jax_weights_flat[i * 2 + 1])
        
        tf_kernel_shape = tf_layer.get_weights()[0].shape
        # JAX (Flax) 的 Dense kernel shape 是 (in_features, out_features)
        # TF Keras 的 Dense kernel shape 也是 (in_features, out_features)
        # 通常不需要轉置，但保留檢查以防萬一
        if jax_kernel.shape != tf_kernel_shape:
            print(f"  [致命錯誤] 層 '{tf_layer.name}' 的權重矩陣形狀不匹配。")
            print(f"    - TF 期望形狀: {tf_kernel_shape}")
            print(f"    - JAX 實際形狀: {jax_kernel.shape}")
            return False
        
        tf_layer.set_weights([jax_kernel, jax_bias])
    return True


def main():
    """主函數"""
    # --- 步驟 0: 配置 ---
    print("\n--- Step 0: Configuration ---")
    env_name = 'PupperJoystickWithGun'
    CHECKPOINT_DIR = epath.Path("checkpoints/" + env_name).resolve()

    # 核心模型的輸入是單步 51 維
    POLICY_OBS_SIZE = 51
    ACTION_SIZE = 12
    POLICY_HIDDEN_LAYER_SIZES = (512, 256, 128)

    # --- 步驟 1: 載入 JAX 參數 ---
    print(f"\n--- Step 1: Loading JAX parameters ---")
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

    # --- 步驟 2: 導出 Normalizer 並創建純 TF 模型 ---
    print("\n--- Step 2: Exporting normalizer and building CORE TF model ---")
    try:
        # 1. 提取並保存 48 維的 mean 和 std
        mean = np.array(normalizer_params.mean['state'])
        std = np.array(normalizer_params.std['state'])
        
        output_dir = epath.Path("onnx")
        output_dir.mkdir(parents=True, exist_ok=True)
        normalizer_output_path = output_dir / f"pupper_ppo_normalizer_{latest_step}.npz"
        np.savez(normalizer_output_path, mean_state=mean, std_state=std)
        print(f"  - Normalizer (48-dim) 已保存至: {normalizer_output_path}")

        # 2. 構建【不含】Normalizer 的 48 維輸入模型
        tf_policy_network = make_tf_core_policy_network(
            policy_obs_size=POLICY_OBS_SIZE,
            act_size=ACTION_SIZE,
            hidden_sizes=POLICY_HIDDEN_LAYER_SIZES,
        )
        print("  - TensorFlow Keras 核心模型 (48-dim input) 構建成功。")
        tf_policy_network.summary()
    except Exception as e:
        print(f"  [錯誤] 構建 TF 模型失敗: {e}")
        raise

    # --- 步驟 3: 轉移權重 ---
    print("\n--- Step 3: Transferring JAX weights to TensorFlow model ---")
    if not transfer_weights(policy_params, tf_policy_network):
        return
    print("  - 權重轉移完成。")

    # --- 步驟 4: 將 TF 模型轉換為 ONNX ---
    print("\n--- Step 4: Converting TensorFlow model to ONNX ---")
    output_path = output_dir / f"pupper_ppo_policy_core_{latest_step}.onnx"
    spec = ({'state': tf.TensorSpec((None, POLICY_OBS_SIZE), tf.float32, name="state")},)
    try:
        model_proto, _ = tf2onnx.convert.from_keras(
            tf_policy_network, 
            input_signature=spec, 
            opset=13,
            output_path=str(output_path)
        )
        print(f"\n  - 轉換成功! ONNX 模型已保存至: {output_path}")
        onnx.checker.check_model(str(output_path))
        print("  - ONNX 模型檢查成功。")
    except Exception as e:
        print(f"  [錯誤] ONNX 轉換過程中發生錯誤: {e}")
        raise

if __name__ == "__main__":
    main()