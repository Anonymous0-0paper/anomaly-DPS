import torch

if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"Total GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
else:
    print("No GPU detected (using CPU)")
print(torch.version.cuda)  # PyTorch's CUDA version

import torch, pynvml
pynvml.nvmlInit()
print("NVML driver version:", pynvml.nvmlSystemGetDriverVersion())
print("CUDA available:", torch.cuda.is_available())
