"""
GPU/CUDA Availability Test Script

This script checks if GPU/CUDA is available for TensorFlow and XGBoost.
Run this before training to verify your GPU setup.
"""

import tensorflow as tf
import xgboost as xgb
import sys
import subprocess

print("="*60)
print("GPU/CUDA Availability Check")
print("="*60)

# Python version
print(f"\nPython version: {sys.version}")
print(f"Python executable: {sys.executable}")

# TensorFlow GPU check
print(f"\n{'TensorFlow GPU Check':-^60}")
print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU devices detected by TensorFlow: {len(gpus)}")
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"    Details: {details}")
        except:
            pass
else:
    print("  [!] No GPU detected by TensorFlow")
    print("  Note: TensorFlow 2.11+ does not support native GPU on Windows")

# XGBoost GPU check
print(f"\n{'XGBoost GPU Check':-^60}")
print(f"XGBoost version: {xgb.__version__}")
try:
    build_info = xgb.build_info()
    use_cuda = build_info.get('USE_CUDA', False)
    print(f"XGBoost CUDA support: {use_cuda}")
    if use_cuda:
        print("  [OK] XGBoost can use GPU acceleration")
    else:
        print("  [!] XGBoost built without CUDA support")
except Exception as e:
    print(f"  [!] Could not determine CUDA support: {e}")

# System CUDA check (nvidia-smi)
print(f"\n{'System CUDA Check (nvidia-smi)':-^60}")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        # Print GPU info lines
        for line in lines[:15]:  # First 15 lines usually contain GPU info
            if line.strip():
                print(f"  {line}")
        print("  [OK] NVIDIA GPU detected by system")
    else:
        print("  [!] nvidia-smi command failed")
        print("  No NVIDIA GPU detected or drivers not installed")
except FileNotFoundError:
    print("  [!] nvidia-smi not found")
    print("  NVIDIA drivers may not be installed")
except Exception as e:
    print(f"  [!] Error running nvidia-smi: {e}")

# Summary
print("\n" + "="*60)
print("Summary")
print("="*60)

tf_gpu = len(gpus) > 0
xgb_cuda = False
try:
    xgb_cuda = xgb.build_info().get('USE_CUDA', False)
except:
    pass

print(f"TensorFlow GPU available: {'YES' if tf_gpu else 'NO (CPU only)'}")
print(f"XGBoost CUDA available: {'YES' if xgb_cuda else 'NO (CPU only)'}")

print("\nTraining will use:")
print(f"  - LSTM (TensorFlow): {'GPU' if tf_gpu else 'CPU'}")
print(f"  - XGBoost: {'GPU' if xgb_cuda else 'CPU'}")

if not tf_gpu and not xgb_cuda:
    print("\n[INFO] CPU training is perfectly fine for 6 months of data!")
    print("Expected training times:")
    print("  - LSTM: ~5-10 minutes")
    print("  - XGBoost: ~1-2 minutes")

print("="*60)
