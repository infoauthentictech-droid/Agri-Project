"""GPU availability check script."""
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available. Training will use CPU (slow).")
    print("To enable GPU:")
    print("  1. Install NVIDIA CUDA Toolkit 12.1")
    print("  2. Install NVIDIA cuDNN 8.9")
    print("  3. Update NVIDIA GPU drivers")
    print("  4. Restart your computer")
