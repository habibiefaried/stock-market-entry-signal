"""
GPU/CUDA Availability Test Script

This script checks if GPU/CUDA is available for TensorFlow and XGBoost.
Run this before training to verify your GPU setup.
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'  # Set PyTorch backend BEFORE importing keras

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

# PyTorch GPU check
print(f"\n{'PyTorch GPU Check':-^60}")
import torch
print(f"PyTorch version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print("  [OK] PyTorch can use GPU acceleration")
else:
    print("  [!] CUDA not available for PyTorch")

# Keras GPU check (with PyTorch backend)
print(f"\n{'Keras + PyTorch Backend Check':-^60}")
import keras
print(f"Keras version: {keras.__version__}")
print(f"Keras backend: {keras.backend.backend()}")
if keras.backend.backend() == 'torch' and cuda_available:
    print("  [OK] Keras using PyTorch backend with GPU!")
else:
    print("  [!] Keras not using GPU")

# TensorFlow check (for reference)
print(f"\n{'TensorFlow Check (Reference)':-^60}")
print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow GPU devices: {len(gpus)}")
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
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

pytorch_gpu = torch.cuda.is_available()
keras_pytorch_gpu = keras.backend.backend() == 'torch' and pytorch_gpu
xgb_cuda = False
try:
    xgb_cuda = xgb.build_info().get('USE_CUDA', False)
except:
    pass

print(f"PyTorch GPU available: {'YES' if pytorch_gpu else 'NO (CPU only)'}")
print(f"Keras + PyTorch GPU available: {'YES' if keras_pytorch_gpu else 'NO (CPU only)'}")
print(f"XGBoost CUDA available: {'YES' if xgb_cuda else 'NO (CPU only)'}")

print("\nTraining will use:")
print(f"  - LSTM (Keras + PyTorch): {'GPU' if keras_pytorch_gpu else 'CPU'}")
print(f"  - XGBoost: {'GPU' if xgb_cuda else 'CPU'}")

if keras_pytorch_gpu and xgb_cuda:
    print("\n[SUCCESS] Both models will use GPU acceleration!")
    print("Expected training times:")
    print("  - LSTM: ~1-2 minutes (GPU)")
    print("  - XGBoost: ~30-60 seconds (GPU)")
elif not keras_pytorch_gpu and not xgb_cuda:
    print("\n[INFO] Both models will use CPU")
    print("Expected training times:")
    print("  - LSTM: ~5-10 minutes")
    print("  - XGBoost: ~1-2 minutes")
else:
    print("\n[INFO] Partial GPU acceleration available")
    if keras_pytorch_gpu:
        print("  - LSTM: ~1-2 minutes (GPU)")
    else:
        print("  - LSTM: ~5-10 minutes (CPU)")
    if xgb_cuda:
        print("  - XGBoost: ~30-60 seconds (GPU)")
    else:
        print("  - XGBoost: ~1-2 minutes (CPU)")

print("="*60)
