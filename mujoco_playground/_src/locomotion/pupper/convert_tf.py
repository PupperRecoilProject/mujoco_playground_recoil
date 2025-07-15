# ===================================================================
#      JAX to ONNX Conversion via TensorFlow (with .npz normalizer export)
# ===================================================================

# --- 導入 ---
import pickle
import numpy as np
import os
import sys
import inspect
from etils import epath
from typing import Any, Dict, Tuple, Sequence

# 強制 JAX 和 TensorFlow 使用 CPU
print("--- Forcing JAX and TensorFlow to use CPU to ensure compatibility ---")
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jp
import tensorflow as tf
from tensorflow.keras import layers
import tf2onnx
import onnx

from orbax import checkpoint as ocp

# --- 將輔助類和函數定義在 main 之外 ---

class NormalizationLayer(layers.Layer):
    """A Keras layer for applying observation normalization."""
    def __init__(self, mean, std, name='normalization_layer', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.std = tf.constant(std, dtype=tf.float32)

    def call(self, inputs):
        state_input = inputs['state']
        return (state_input - self.mean) / (self.std + 1e-8)

class KerasMLP(tf.keras.Model):
    """A Keras model equivalent to Brax's MLP."""
    def __init__(self, layer_sizes, activation=tf.nn.swish, name='mlp_block'):
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

def make_tf_policy_network(policy_obs_size, act_size, hidden_sizes, normalizer_mean, normalizer_std):
    """Creates the complete, end-to-end TensorFlow policy network."""
    inputs = {'state': tf.keras.Input(shape=(policy_obs_size,), name='state')}
    normalized_obs = NormalizationLayer(normalizer_mean, normalizer_std)(inputs)
    mlp = KerasMLP(layer_sizes=list(hidden_sizes) + [act_size * 2])
    logits = mlp(normalized_obs)
    loc = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1)[0])(logits)
    outputs = tf.keras.layers.Activation('tanh', name='action')(loc)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="PupperPPOPolicy")

def transfer_weights(jax_policy_params, tf_model):
    """Transfers weights from a JAX parameter pytree to a TensorFlow model."""
    tf_mlp_layers = [l for l in tf_model.get_layer('mlp_block').layers if isinstance(l, layers.Dense)]
    jax_weights_flat = jax.tree_util.tree_leaves(jax_policy_params)

    if len(tf_mlp_layers) * 2 != len(jax_weights_flat):
        print("  [FATAL ERROR] Mismatch between number of TF layers and JAX weights!")
        return False
    
    for i, tf_layer in enumerate(tf_mlp_layers):
        jax_bias = np.array(jax_weights_flat[i * 2])
        jax_kernel = np.array(jax_weights_flat[i * 2 + 1])
        
        tf_kernel_shape = tf_layer.get_weights()[0].shape
        if jax_kernel.shape != tf_kernel_shape:
            if jax_kernel.T.shape == tf_kernel_shape:
                jax_kernel = jax_kernel.T
            else:
                print(f"  [FATAL ERROR] Kernel shape mismatch for layer '{tf_layer.name}'.")
                return False
        
        tf_layer.set_weights([jax_kernel, jax_bias])
    return True


def main():
    """Main function body."""
    # --- 步驟 0: 配置 ---
    print("\n--- Step 0: Configuration ---")
    env_name = 'PupperJoystickFlatTerrain' # 我們將從這個環境的 checkpoint 中提取
    CHECKPOINT_DIR = epath.Path("checkpoints/" + env_name).resolve() 
    POLICY_OBS_SIZE = 48 
    ACTION_SIZE = 12
    POLICY_HIDDEN_LAYER_SIZES = (512, 256, 128)

    # --- 步驟 1: 載入 JAX 參數 ---
    print(f"\n--- Step 1: Loading JAX parameters from pickle file ---")
    latest_step = None
    try:
        steps = [int(p.name) for p in CHECKPOINT_DIR.iterdir() if p.is_dir() and p.name.isdigit()]
        latest_step = 30965760 # max(steps) if steps else None
        pkl_path = CHECKPOINT_DIR / str(latest_step) / "params.pkl"
        print(f"  - Loading from: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            params_tuple = pickle.load(f)
        normalizer_params, policy_params, _ = params_tuple
        print("  - JAX parameters loaded and separated successfully.")
    except Exception as e:
        print(f"  [FATAL ERROR] Could not load parameters: {e}")
        raise

    # --- 步驟 2: 創建 TensorFlow Keras 模型 (同時提取和保存 Normalizer 參數) ---
    print("\n--- Step 2: Building TF model and exporting normalizer params ---")
    try:
        mean = np.array(normalizer_params.mean['state'])
        std = np.array(normalizer_params.std['state'])
        print("  - Mean and Std extracted successfully.")

        '''# ===================================================================
        #           【關鍵新增功能】: 保存 mean 和 std 到 .npz 檔案
        # ===================================================================
        normalizer_output_path = f"pupper_ppo_normalizer_{latest_step}.npz"
        np.savez(
            normalizer_output_path,
            mean_state=mean,
            std_state=std
        )
        print(f"  - Normalizer parameters saved to: {normalizer_output_path}")
        # ==================================================================='''

        tf_policy_network = make_tf_policy_network(
            policy_obs_size=POLICY_OBS_SIZE,
            act_size=ACTION_SIZE,
            hidden_sizes=POLICY_HIDDEN_LAYER_SIZES,
            normalizer_mean=mean,
            normalizer_std=std
        )
        print("  - TensorFlow Keras model built successfully.")
        tf_policy_network.summary()
    except Exception as e:
        print(f"  [ERROR] Failed to build TF model: {e}")
        raise

    # --- 步驟 3: 轉移權重 ---
    print("\n--- Step 3: Transferring JAX weights to TensorFlow model ---")
    if not transfer_weights(policy_params, tf_policy_network):
        return
    print("  - Weight transfer complete.")

    # --- 步驟 4: 將 TF 模型轉換為 ONNX ---
    print("\n--- Step 4: Converting TensorFlow model to ONNX ---")
    output_path = f"pupper_ppo_policy_{latest_step}_tf_converted.onnx"
    spec = ({'state': tf.TensorSpec((None, POLICY_OBS_SIZE), tf.float32, name="state")},)
    try:
        model_proto, _ = tf2onnx.convert.from_keras(
            tf_policy_network, 
            input_signature=spec, 
            opset=13,
            output_path=output_path
        )
        print(f"\n  - Conversion successful! ONNX model saved to: {output_path}")
        onnx.checker.check_model(output_path)
        print("  - ONNX model checked successfully.")
    except Exception as e:
        print(f"  [ERROR] An error occurred during ONNX conversion: {e}")
        raise

if __name__ == "__main__":
    main()