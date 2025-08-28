# ===================================================================
#             ONNX 模型純驗證腳本 (最終核心驗證版)
# ===================================================================
import pickle
import jax
import jax.numpy as jp
import numpy as np
import onnx
import onnxruntime as ort
from etils import epath
import sys
import traceback
from typing import Dict, Tuple, Sequence

# --- 【新增導入】---
from flax import linen as nn

# --- 將專案根目錄添加到 Python 路徑 ---
try:
    project_root = epath.Path(__file__).parent.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
except NameError:
    print("警告: __file__ 未定義。假設專案已在 PYTHONPATH 中。")
    pass

# --- 【核心】手動定義一個與 Brax MLP 兼容的 Flax 模型 ---
class SimpleMLP(nn.Module):
    layer_sizes: Sequence[int]
    activation: nn.activation = nn.swish

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.layer_sizes):
            act = self.activation if i < len(self.layer_sizes) - 1 else None
            x = nn.Dense(
                features=size,
                kernel_init=jax.nn.initializers.lecun_uniform(),
                name=f'hidden_{i}'
            )(x)
            if act:
                x = act(x)
        return x

def main():
    """主驗證函數"""

    # --- 步驟 1: 配置 ---
    print("\n--- Step 1: Configuration ---")
    env_name = 'PupperJoystickWithGun'
    # 自動查找最新的 checkpoint
    script_dir = epath.Path(__file__).parent
    checkpoint_dir = script_dir / f"checkpoints/{env_name}"
    steps = [int(p.name) for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    latest_step = 101580800 # max(steps) if steps else None
    if not latest_step:
        print(f"致命錯誤: 在 {checkpoint_dir} 中找不到任何 checkpoint。")
        return
    step = str(latest_step)
    
    pkl_path = checkpoint_dir / step / "params.pkl"
    onnx_model_path = script_dir / f"onnx/pupper_ppo_policy_core_{step}.onnx"
    normalizer_path = script_dir / f"onnx/pupper_ppo_normalizer_{step}.npz"
    
    POLICY_OBS_SIZE = 51
    ACTION_SIZE = 12
    POLICY_HIDDEN_LAYER_SIZES = (512, 256, 128)
    
    print(f"JAX 參數路徑: {pkl_path}")
    print(f"ONNX 模型路徑: {onnx_model_path}")
    print(f"Normalizer 參數路徑: {normalizer_path}")

    # --- 步驟 2: 加載所有必要的參數 ---
    print("\n--- Step 2: Loading all parameters ---")
    try:
        with open(pkl_path, 'rb') as f:
            params = pickle.load(f)
        normalizer_params, policy_params, _ = params
        
        normalizer_data = np.load(normalizer_path)
        mean_48d = normalizer_data['mean_state']
        std_48d = normalizer_data['std_state']

        print("JAX 和 Normalizer 參數載入成功。")
    except Exception as e:
        print(f"致命錯誤: 加載參數失敗: {e}")
        traceback.print_exc()
        return

    # --- 步驟 3: 手動重建 JAX 推理函數 ---
    print("\n--- Step 3: Manually reconstructing the JAX inference function ---")
    policy_network = SimpleMLP(
        layer_sizes=list(POLICY_HIDDEN_LAYER_SIZES) + [ACTION_SIZE * 2]
    )

    @jax.jit
    def jit_inference_fn(obs_single_step, norm_mean, norm_std, policy_p):
        normalized_obs = (obs_single_step - norm_mean) / (norm_std + 1e-8)
        logits = policy_network.apply(policy_p, normalized_obs)
        loc, _ = jp.split(logits, 2, axis=-1)
        action = jp.tanh(loc)
        return action
    
    print("JAX 推理函數手動重建成功。")

    # --- 步驟 4: 準備完全一致的 48 維測試數據 ---
    print("\n--- Step 4: Preparing identical 48-dim test data ---")
    key_for_test = jax.random.PRNGKey(42)
    raw_obs_jax = jax.random.normal(key_for_test, (1, POLICY_OBS_SIZE), dtype=jp.float32)
    raw_obs_numpy = np.array(raw_obs_jax)

    # --- 步驟 5: 運行 JAX 和 ONNX 模型 ---
    print("\n--- Step 5: Running both models ---")
    
    # 運行 JAX
    print("使用 JAX 模型進行推理...")
    jax_output = jit_inference_fn(raw_obs_jax, mean_48d, std_48d, policy_params)
    jax_output_np = np.array(jax_output)
    print(f"JAX 模型輸出 (動作):\n{jax_output_np}")

    # 運行 ONNX
    print("\n使用 ONNX 模型進行推理...")
    onnx_output = None
    try:
        normalized_obs_numpy = (raw_obs_numpy - mean_48d) / (std_48d + 1e-8)
        
        ort_session = ort.InferenceSession(str(onnx_model_path))
        onnx_inputs = {'state': normalized_obs_numpy}
        onnx_output = ort_session.run(None, onnx_inputs)[0]
        print(f"ONNX 模型輸出 (動作):\n{onnx_output}")
    except Exception as e:
        print(f"致命錯誤: 運行 ONNX 推理失敗: {e}")
        traceback.print_exc()

    # --- 步驟 6: 比較結果 ---
    print("\n--- Step 6: Comparing JAX and ONNX outputs ---")
    if jax_output_np is not None and onnx_output is not None:
        try:
            np.testing.assert_allclose(jax_output_np, onnx_output, rtol=1e-5, atol=1e-5)
            print("\n✅ 成功: ONNX 模型輸出與 JAX 模型輸出完美匹配！")
        except AssertionError as e:
            print("\n❌ 失敗: ONNX 模型輸出與 JAX 模型輸出不匹配！")
            print("詳細差異:")
            print(e)

if __name__ == '__main__':
    main()