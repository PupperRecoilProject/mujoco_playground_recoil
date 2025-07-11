# ===================================================================
#             ONNX 模型純驗證腳本 (最終完整版)
# ===================================================================
import pickle
import jax
import jax.numpy as jp
import numpy as np
import onnx
import onnxruntime as ort
import functools
from etils import epath
import sys
import traceback
from typing import Dict, Tuple

# 將專案根目錄添加到 Python 路徑，以確保導入成功
# 這是一個健壯的做法，避免因運行位置不同而導致的 ImportError
try:
    # 假設 test_onnx.py 位於 .../locomotion/pupper/
    # 我們需要添加 `mujoco_playground_recoil` 這個根目錄
    project_root = epath.Path(__file__).parent.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
except NameError:
    # 如果在 Notebook 中以單元格形式運行，__file__ 未定義，
    # 這種情況下，請確保您的 Notebook 環境已經能正確導入 mujoco_playground
    print("Warning: __file__ not defined. Assuming project is in PYTHONPATH.")
    pass

# --- 步驟 1: 導入所有必要的模組 ---
print("--- Step 1: Importing modules ---")
try:
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground.config import locomotion_params
    from brax.training.acme import running_statistics
    from mujoco_playground import registry
except ImportError as e:
    print(f"FATAL ERROR: Error importing necessary modules: {e}")
    print("Please ensure 'brax' and 'mujoco_playground' are correctly installed "
          "and their parent directory is in your PYTHONPATH.")
    exit()

def main():
    """Main function to run the validation."""

   # ===================================================================
    #      步驟 2: 配置 - 指定檔案路徑 (最終穩健版)
    # ===================================================================
    print("\n--- Step 2: Configuration ---")
    env_name = 'PupperJoystickFlatTerrain'
    step = '200540160' 
    
    # 【關鍵修改】: 使用 __file__ 來獲取腳本所在的目錄
    # 這使得路徑解析與您在哪裡運行命令無關，更加穩健
    script_dir = epath.Path(__file__).parent
    
    # --- 指向 JAX 的 .pkl 參數檔案 ---
    # 從腳本所在目錄開始構建路徑
    pkl_path = script_dir / f"checkpoints/{env_name}/{step}/params.pkl"
    
    # --- 指向 ONNX 模型檔案 ---
    # 從腳本所在目錄開始構建路徑
    onnx_model_path = script_dir / f"pupper_ppo_policy_{step}_tf_converted.onnx"

    print(f"Attempting to load JAX parameters from: {pkl_path.as_posix()}")
    print(f"Attempting to load ONNX model from: {onnx_model_path.as_posix()}")

    # 【關鍵】: 直接定義觀測規格，不再動態獲取
    print("\n--- Defining Observation Specification ---")
    obs_spec: Dict[str, Tuple[int, ...]] = {
        'state': (48,),
        'privileged_state': (123,) 
    }
    action_size = 12
    print(f"Using hardcoded observation spec: {obs_spec}")
    print(f"Action size: {action_size}")

    # ===================================================================
    #      步驟 3: 加載 JAX 參數並重建 JAX 推理函數
    # ===================================================================
    print("\n--- Step 3: Reconstructing the JAX inference function ---")
    try:
        with open(pkl_path, 'rb') as f:
            # params 是一個元組: (normalizer_params, policy_params, value_params)
            params = pickle.load(f)
        print("Parameters loaded successfully.")

        # 獲取與訓練時完全相同的網路配置
        ppo_params = locomotion_params.brax_ppo_config(env_name)
        network_config = ppo_params.network_factory
        
        # 重建 PPO 網路
        ppo_network = ppo_networks.make_ppo_networks(
            observation_size=obs_spec,
            action_size=action_size,
            preprocess_observations_fn=running_statistics.normalize,
            **network_config
        )

        # 使用 Brax 的標準工廠函數來創建推理函數
        make_inference_fn_factory = ppo_networks.make_inference_fn(ppo_network)
        
        # 使用加載的 params 創建最終的、帶有正確參數綁定的推理函數
        inference_fn_with_key = make_inference_fn_factory(params, deterministic=True)
        
        # JIT 編譯以獲取優化的計算圖
        jit_inference_fn_with_key = jax.jit(inference_fn_with_key)
        
        # 為了方便呼叫，創建一個只接收 obs 的版本
        def final_jit_inference_fn(obs_dict):
            actions, _ = jit_inference_fn_with_key(obs_dict, jax.random.PRNGKey(0))
            return actions

        print("JAX inference function reconstructed successfully.")

    except FileNotFoundError:
        print(f"FATAL ERROR: JAX checkpoint file not found at {pkl_path}.")
        return
    except Exception:
        print(f"FATAL ERROR: Failed to reconstruct the JAX function.")
        traceback.print_exc()
        return

    # ===================================================================
    #      步驟 4: 準備驗證數據並運行 JAX 模型 (獲取標準答案)
    # ===================================================================
    print("\n--- Step 4: Preparing test data and running JAX model ---")
    
    key_for_test = jax.random.PRNGKey(42)
    
    # 使用 is_leaf 參數告訴 tree_map 不要深入元組
    test_obs_dict_jax = jax.tree_util.tree_map(
        lambda shape_tuple: jax.random.normal(key_for_test, (1,) + shape_tuple, dtype=jp.float32), 
        obs_spec,
        is_leaf=lambda x: isinstance(x, tuple)
    )
    test_obs_dict_numpy = {
        key: np.array(value) for key, value in test_obs_dict_jax.items()
    }

    print("Running inference with JAX model...")
    jax_output = final_jit_inference_fn(test_obs_dict_jax)
    jax_output_np = np.array(jax_output)
    print(f"JAX model output (action):\n{jax_output_np}")

    # ===================================================================
    #      步驟 5: 加載並運行 ONNX 模型 (最終修正版)
    # ===================================================================
    print("\n--- Step 5: Loading and running ONNX model ---")
    
    onnx_output = None
    try:
        if not onnx_model_path.exists():
            print(f"ERROR: ONNX model file not found at {onnx_model_path}.")
        else:
            ort_session = ort.InferenceSession(str(onnx_model_path))
            
            # 【關鍵修改】: 創建一個只包含 ONNX 模型所需輸入的字典
            # 我們知道它只需要 'state'
            onnx_inputs = {
                'state': test_obs_dict_numpy['state']
            }

            # 獲取並打印模型的真實輸入/輸出名稱，以便 debug
            input_names = [inp.name for inp in ort_session.get_inputs()]
            output_names = [out.name for out in ort_session.get_outputs()]
            print(f"ONNX model expected input names: {input_names}")
            print(f"ONNX model output names: {output_names}")

            # 使用這個新的、更簡潔的輸入字典進行推理
            onnx_output = ort_session.run(None, onnx_inputs)[0]
            print(f"ONNX model output shape: {onnx_output.shape}")

    except Exception as e:
        print(f"FATAL ERROR: Failed to run ONNX inference: {e}")
        traceback.print_exc()

    # ===================================================================
    #      步驟 6: 比較 JAX 和 ONNX 的輸出結果
    # ===================================================================
    print("\n--- Step 6: Comparing JAX and ONNX outputs ---")
    
    if jax_output_np is not None and onnx_output is not None:
        try:
            np.testing.assert_allclose(jax_output_np, onnx_output, rtol=1e-5, atol=1e-5)
            print("\n✅ SUCCESS: ONNX model output perfectly matches JAX model output!")
        except AssertionError as e:
            print("\n❌ FAILURE: ONNX model output DOES NOT match JAX model output!")
            print("Detailed difference:")
            print(e)
    else:
        print("\n❌ FAILURE: Could not compare outputs because one of the models failed to produce an output.")

if __name__ == '__main__':
    main()