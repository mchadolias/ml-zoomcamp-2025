<<<<<<< HEAD
# **Complete PyTorch Installation Guide (GPU & CPU Support)**

Hi 👋  
This guide walks you through installing PyTorch with GPU acceleration. It covers multiple package managers (uv, pip, conda), explains CUDA compatibility, and includes verification and benchmarking scripts.

Tested on Linux and WSL2.

---

## **Table of Contents**

1. [Prerequisites](#prerequisites)
2. [Step-by-Step Installation](#step-by-step-installation)
   - [Step 1: Check Your Graphics Card](#step-1-check-your-graphics-card)
   - [Step 2: Choose and Setup Package Manager](#step-2-choose-and-setup-package-manager)
   - [Step 3: Install PyTorch](#step-3-install-pytorch)
   - [Step 4: Verify Installation](#step-4-verify-installation)
3. [GPU vs CPU](#why-gpu-over-cpu)
4. [Troubleshooting Common Issues](#troubleshooting-common-issues)
5. [Best Practices & Optimization Tips](#best-practices--optimization-tips)
6. [FAQs](#faqs)

---

## **Prerequisites**

Before installing PyTorch, ensure your system meets these requirements:

### **System Requirements**

- **OS**: Linux, WSL2 (recommended for Windows), or macOS
- **Shell**: bash, zsh, or similar
- **Python**: 3.9 or later (always check beforehand)
- **Memory**: Minimum 8 GB RAM, 16+ GB recommended for deep learning

### **GPU-Specific Requirements**

- **NVIDIA GPU**: Compute capability ≥ 3.5 (most GPUs from 2014+)
- **NVIDIA Driver**: Latest version recommended
- **AMD GPU**: ROCm support (check compatibility)
- **Integrated Graphics**: CPU-only mode available but not recommended for training

### **Package Managers**

Choose one based on your needs:

- **uv** ⚡ (Recommended): Fast, modern, reliable
- **pip**: Universal, simple
- **conda**: Excellent dependency management
- **pipenv**: Good for application deployment

> **📝 Note for Seminar Participants**: This seminar recommends **uv** for its speed and reliability.

---

## **Step-by-Step Installation**

### **Step 1: Check Your Graphics Card**

The first step is to identify your GPU's compute capabilities. This determines whether you can use GPU acceleration and which CUDA version to choose.

**For NVIDIA GPUs:**
=======
# Installation guide for PyTorch

Hi 👋
This is a small guide to how you can install PyTorch to your PC, with or without GPU support. This might serve for the deep-learning part of the module, but for deep learning activities in general. This guide has been tested in a Linux-based subsystem, but in principle should work in WSL as well.

## Prerequisites

- Linux OS or WSL
- bash/zsh shell
- Python
- Package manager (`pip`)
- Python manager (`uv`, `conda`, `pipenv`, etc)

## prerequisites

- Linux OS or WSL
- bash/zsh shell
- Python
- Package manager (`pip`)
- Python manager (`uv`, `conda`, `pipenv`, etc)

## Step Guide

### Step 1: Check your Graphics Card

The first part is to identify the compute capabilities of the GPU that your system is using.

For systems that are using an NVIDIA GPU, using the `nvidia-smi` command lets the user find out valuable information regarding the Graphics Cards and its software. For us, it's useful to check the CUDA version that your system is using and the model name of the card.
>>>>>>> upstream/master

```bash
nvidia-smi
```

Example output (shortened):

<<<<<<< HEAD
```shell
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08   Driver Version: 580.105.08   CUDA Version: 13.0     |
|   0  NVIDIA GeForce GTX 1650         On | 00000000:01:00.0 On |         N/A |
+-----------------------------------------------------------------------------+
```

In this example:

- `Driver Version`: 580.105.08
- `CUDA Version` (max supported): 13.0
- `GPU Model`: NVIDIA GeForce GTX 1650

**Important Notes:**

1. The CUDA version shown is the maximum version your driver supports, not what you need to install
2. PyTorch bundles its own CUDA toolkit - you don't need to install CUDA separately
3. It's generally safe to use the latest CUDA version that PyTorch supports

**Key information to note:**

- **Driver Version**: Should be ≥ 525 for CUDA 12.x
- **CUDA Version**: Maximum CUDA version your driver supports
- **GPU Model**: Check compute capability

**Check compute capability:**
=======
```
Sat Nov 29 19:40:41 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA GeForce GTX 1650        Off |   00000000:01:00.0  On |                  N/A |
| N/A   47C    P8              2W /   50W |    1173MiB /   4096MiB |     16%      Default |
+-----------------------------------------+------------------------+----------------------+
```

For my case as listed above it is `CUDA Version: 13.0` and `Model name: NVIDIA GeForce GTX 1650`

**Important**: The CUDA version shown in `nvidia-smi` is the maximum version your driver supports, but PyTorch will use its own CUDA toolkit. It's generally safe to use the latest CUDA version that PyTorch supports.

Copy the name of your graphics card to find out its capabilities. For PyTorch to be effective with a graphics card, it needs to have a compute capability of 3.5 or higher. Most modern GPUs (2014+) meet this requirement. The GTX 1650 you listed has compute capability 7.5, which is excellent.

This can be easily checked from lookup tables. For instance, NVIDIA provides this one for its [models](https://developer.nvidia.com/cuda-gpus).

**Alternative method to check compute capability:**
>>>>>>> upstream/master

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

<<<<<<< HEAD
You can check specific GPU capabilities on NVIDIA's [CUDA GPU list](https://developer.nvidia.com/cuda-gpus).

**For AMD GPUs:**

```bash
amd-smi
```

AMD uses its own framework called ROCm. Check compatibility at the AMD GPU [Architecture Table](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html).

**For Intel/Integrated Graphics:**
You'll use CPU-only installation.

> **⚠️ Important**: The CUDA version in `nvidia-smi` is the **maximum supported**, not what you need to install. PyTorch bundles its own CUDA toolkit.

### **Step 2: Choose and Setup Package Manager**

#### **Option A: Using uv (Recommended)**

**Install uv:**

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: via pip
pip install uv

# Verify installation
uv --version
```

**Create virtual environment:**

```bash
# Create environment
uv venv pytorch-env

# Activate it
source pytorch-env/bin/activate  # Linux/macOS
# OR
.\pytorch-env\Scripts\activate   # Windows
```

#### **Option B: Using pip**

**Create virtual environment:**

```bash
python -m venv pytorch-env
source pytorch-env/bin/activate  # Linux/macOS
.\pytorch-env\Scripts\activate   # Windows
```

#### **Option C: Using conda**

**Install Miniconda/Anaconda, then:**

```bash
conda create -n pytorch-env python=3.10
conda activate pytorch-env
```

### **Step 3: Install PyTorch**

PyTorch provides different versions for different CUDA versions, AMD ROCm, and CPU-only setups. Currently supported versions include CUDA 12.6, 12.8, 13.0, and ROCm 6.4.

#### **CUDA Version Compatibility Guide**

| Your Driver Shows | Recommended PyTorch CUDA | Reason |
|------------------|--------------------------|--------|
| CUDA 11.x | CUDA 11.8 | Maximum compatibility |
| CUDA 12.0-12.5 | CUDA 12.1 | Stable, widely supported |
| CUDA 12.6+ | CUDA 12.8 | Latest stable |
| CUDA 13.0+ | CUDA 12.8 or 13.0 | 12.8 is more stable |
| No GPU/Other | CPU | For inference/small models |

#### **Installation Commands**

**Using uv:**

```bash
# For CUDA 12.8 (recommended for most users)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# For CUDA 12.6
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# For AMD ROCm 6.4
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# For CPU-only
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Using pip:**

```bash
# CUDA 12.8 (default as of PyTorch 2.3+)
pip install torch torchvision torchaudio

# Specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Using conda:**

```bash
# CUDA 12.8
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia

# CPU-only
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

If you encounter Python version conflicts, you can always downgrade. Use:

```bash
# Downgrade Python if needed
conda create -n pytorch-env python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia
```

### **Step 4: Verify Installation**

Run this simple verification script to confirm PyTorch is installed correctly and can access your GPU:

**Basic Verification Script (`verify_basic.py`):**

```python
import torch
import torchvision

print("=" * 50)
print("PyTorch Installation Verification")
print("=" * 50)

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"\nGPU Information:")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU Count: {torch.cuda.device_count()}")
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  Memory: {memory_gb:.1f} GB")
    print(f"  Compute Capability: {torch.cuda.get_device_capability()}")
else:
    print("\nRunning in CPU-only mode")
    
print("=" * 50)
```

**Run the verification:**

```bash
uv run verify_basic.py
# or
python verify_basic.py
```

**Expected output (GPU available):**

```shell
==================================================
PyTorch Installation Verification
==================================================
PyTorch version: 2.9.1+cu126
Torchvision version: 0.24.1+cu126
CUDA available: True

GPU Information:
  Device: NVIDIA GeForce GTX 1650
  CUDA version: 12.6
  GPU Count: 1
  Memory: 3.6 GB
  Compute Capability: (7, 5)
==================================================
```

---

## **GPU vs CPU**

GPUs are strongly recommended over CPUs for deep learning because they can perform thousands of calculations simultaneously, while CPUs process tasks sequentially. This parallel processing capability is perfect for the matrix operations that form the foundation of neural networks.

**Prove it to yourself:** Run the benchmark script below to see exactly how much faster your GPU performs compared to CPU!

**File: `benchmark_gpu_cpu.py`**
=======
Similarly, for AMD Graphics cards the `amd-smi` command exists, with the following [table](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html). It also uses its own framework called `ROCm`.

### Step 2: Visit the PyTorch docs

After the user has identified the system, it is time to visit the PyTorch documentation guide, where the exact capabilities of the system can be filled for the appropriate version of the framework to be provided. Here the user can decide to run deep learning only on its CPU for various reasons. However, generally it is not recommended for larger models.

**Note:** Remember to install the library inside your python environment to avoid conflicts with your system Python.

```bash
conda activate deep-learning-env # Or something similar for other managers
```

**Package Manager Note**: While `pip` works well, Conda often handles CUDA dependencies better and is recommended for easier dependency management.

As well as, PyTorch is provided apart from Python, also as a library for C++/Java called `libtorch`, though I haven't used it in that capacity so I can't provide any comment.

Currently, PyTorch is provided for CUDA 12.6, 12.8 and 13.0, as well as a separate AMD version, and the CPU-only option. Here are the following links for each option:

**CUDA 12.6 system:**

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**CUDA 12.8 system:**

```bash
pip3 install torch torchvision
```

**CUDA 13.0 system:**

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**AMD (ROCm 6.4) system:**

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

**CPU-only option (Not recommended for training):**

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Verify that Pytoch is working with GPU

A simple script can be used as a test, to verify the proper installation of PyToch to your system. I have provided the one that I used personally when working for the module. 

```python
import torch
import sys

print("=" * 50)
print("Framework GPU Verification")
print("=" * 50)

# System and Python info
print(f"Python version: {sys.version}")
print()

# PyTorch Info
print("\nPYTORCH:")
print(f"  Version: {torch.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")
print(f"  CUDA Version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Count: {torch.cuda.device_count()}")

    # Test PyTorch GPU computation
    device = torch.device("cuda")
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = x + y
    print(f"  GPU Test: Computation successful on {z.device}")
else:
    print("  GPU Test: Using CPU")
```

The output should be something similar to that

```bash
==================================================
Framework GPU Verification
==================================================
Python version: 3.13.7 | packaged by conda-forge | (main, Sep  3 2025, 14:30:35) [GCC 14.3.0]

PYTORCH:
  Version: 2.9.1+cu126
  CUDA Available: True
  CUDA Version: 12.6
  GPU Device: NVIDIA GeForce GTX 1650
  GPU Count: 1
  GPU Test: Computation successful on cuda:0
  Performance Test: 29.89 ms
==================================================
```

## Why Use GPU for Deep Learning?

GPUs are strongly recommended over CPUs for deep learning because they can perform thousands of calculations simultaneously, while CPUs process tasks sequentially. This parallel processing capability is perfect for the matrix operations that form the foundation of neural networks. In practice, this means:

- **Faster training**: Models that take days on CPU might take only hours on GPU
- **Practical workflows**: Makes experimenting with different models feasible
- **Better for larger models**: Handles complex architectures that would be too slow on CPU

The performance difference is substantial - typically 10x to 50x speed improvements for common deep learning tasks.

**Prove it to yourself**: Run the verification script in this guide to see exactly how much faster your GPU performs compared to CPU!
>>>>>>> upstream/master

```python
import torch
import sys
import time

print("=" * 60)
print("PyTorch GPU Verification & Performance Benchmark")
print("=" * 60)

# System and Python info
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print()

# PyTorch Info
print("SYSTEM INFORMATION:")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Count: {torch.cuda.device_count()}")
<<<<<<< HEAD
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
=======
    print(
        f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
>>>>>>> upstream/master
    print(f"  Compute Capability: {torch.cuda.get_device_capability()}")
else:
    print("  No GPU detected - using CPU only")

print("\n" + "=" * 60)
print("PERFORMANCE BENCHMARK")
print("=" * 60)

<<<<<<< HEAD
=======

>>>>>>> upstream/master
def benchmark_operation(operation_name, operation, device, size=1000, iterations=100):
    """Benchmark a single operation on specified device"""
    # Warm-up
    for _ in range(10):
        _ = operation(device, size)

    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        result = operation(device, size)
    end_time = time.time()

    # Sync if CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    return (end_time - start_time) * 1000  # Convert to milliseconds

<<<<<<< HEAD
=======

>>>>>>> upstream/master
# Define benchmark operations
def matmul_operation(device, size=1000):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    return torch.matmul(a, b)

<<<<<<< HEAD
=======

>>>>>>> upstream/master
def elementwise_operation(device, size=1000):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    return a * b + torch.sin(a) - torch.cos(b)

<<<<<<< HEAD
=======

>>>>>>> upstream/master
def convolution_operation(device, size=1000):
    # Smaller size for conv to avoid memory issues
    channels = 32
    spatial = min(size // 8, 128)
    x = torch.randn(1, channels, spatial, spatial, device=device)
    conv = torch.nn.Conv2d(channels, channels, 3, padding=1, device=device)
    return conv(x)

<<<<<<< HEAD
=======

>>>>>>> upstream/master
# Available devices
devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda"))

# Run benchmarks
operations = [
    ("Matrix Multiplication (1000x1000)", matmul_operation),
    ("Element-wise Operations", elementwise_operation),
    ("Convolution Operation", convolution_operation),
]

results = {}

for op_name, op_func in operations:
    print(f"\n{op_name}:")
    results[op_name] = {}

    for device in devices:
        try:
            time_taken = benchmark_operation(op_name, op_func, device)
            results[op_name][device.type] = time_taken
            print(f"  {device.type.upper():<6}: {time_taken:8.2f} ms")
        except RuntimeError as e:
            print(f"  {device.type.upper():<6}: Failed - {str(e)}")
            results[op_name][device.type] = None

# Calculate speedup
print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)

if torch.cuda.is_available():
    for op_name in results:
        cpu_time = results[op_name].get("cpu")
        cuda_time = results[op_name].get("cuda")

        if cpu_time and cuda_time and cuda_time > 0:
            speedup = cpu_time / cuda_time
            print(f"{op_name}:")
            print(f"  Speedup: {speedup:6.1f}x faster on GPU")
            if speedup > 1:
                print(f"  GPU is {speedup:.1f}x faster than CPU")
            else:
                print(f"  CPU is {1/speedup:.1f}x faster than GPU")
        else:
            print(f"{op_name}: Comparison not available")

    # Memory benchmark
    print(f"\nMEMORY BENCHMARK:")
    try:
        # Test large tensor allocation
        large_tensor_gpu = torch.randn(5000, 5000, device=torch.device("cuda"))
        gpu_memory_usage = (
            large_tensor_gpu.element_size() * large_tensor_gpu.nelement() / 1024**2
        )
        print(f"  Allocated {gpu_memory_usage:.1f} MB on GPU successfully")
        del large_tensor_gpu
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"  GPU memory test failed: {e}")

# Final verification
print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

if torch.cuda.is_available():
    # Test basic GPU functionality
    device = torch.device("cuda")

    # Simple computation test
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    z = x + y

    # Backward pass test
    w = torch.randn(10, 10, device=device, requires_grad=True)
    loss = w.sum()
    loss.backward()

    print("✓ Basic GPU computation: SUCCESS")
    print("✓ GPU gradient computation: SUCCESS")
    print(f"✓ Final tensor device: {z.device}")

    # Memory info
    allocated = torch.cuda.memory_allocated() / 1024**2
    cached = torch.cuda.memory_reserved() / 1024**2
    print(f"✓ GPU memory allocated: {allocated:.1f} MB")
    print(f"✓ GPU memory cached: {cached:.1f} MB")

    print("\n🎉 PyTorch GPU setup is working correctly!")
else:
    print("⚠️  Running in CPU-only mode")
    print("💡 For GPU acceleration, check your CUDA installation and drivers")

print("=" * 60)
```

<<<<<<< HEAD
**Example Output (shortened):**

```shell
============================================================
PERFORMANCE BENCHMARK
============================================================

Matrix Multiplication (1000x1000):
  CPU   :  1631.63 ms
  CUDA  :     3.23 ms

Element-wise Operations:
  CPU   :  1188.86 ms
  CUDA  :     4.86 ms

Convolution Operation:
  CPU   :   342.91 ms
  CUDA  :    14.57 ms
=======
After running the script for my systeml, I got the following results 🎉


```bash
============================================================
PyTorch GPU Verification & Performance Benchmark
============================================================
Python version: 3.13.7 | packaged by conda-forge | (main, Sep  3 2025, 14:30:35) [GCC 14.3.0]
PyTorch version: 2.9.1+cu126

SYSTEM INFORMATION:
  CUDA Available: True
  CUDA Version: 12.6
  GPU Device: NVIDIA GeForce GTX 1650
  GPU Count: 1
  GPU Memory: 3.6 GB
  Compute Capability: (7, 5)
>>>>>>> upstream/master

============================================================
PERFORMANCE BENCHMARK
============================================================

Matrix Multiplication (1000x1000):
<<<<<<< HEAD
  CPU   :  1631.63 ms
  CUDA  :     3.23 ms

Element-wise Operations:
  CPU   :  1188.86 ms
  CUDA  :     4.86 ms

Convolution Operation:
  CPU   :   342.91 ms
  CUDA  :    14.57 ms
```

---

## **Troubleshooting Common Issues**

### **Issue 1: CUDA Not Available**

```bash
# Check if PyTorch sees CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Solution steps:
1. Update NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
2. Verify GPU compute capability: nvidia-smi --query-gpu=compute_cap --format=csv
3. Reinstall PyTorch with correct CUDA version
4. For WSL2: Ensure GPU passthrough is enabled
```

### **Issue 2: Out of Memory (OOM)**

```python
# Reduce batch size
batch_size = 32  # Try 16, 8, or 4

# Use gradient accumulation
accumulation_steps = 4
loss.backward()
if (batch_idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### **Issue 3: Python Version Conflicts**

```bash
# With uv: Specify Python version during env creation
uv venv --python 3.10 pytorch-env

# With conda: Specify Python in conda create
conda create -n pytorch-env python=3.10 pytorch torchvision torchaudio

# With pip: Use specific Python version
python3.10 -m venv pytorch-env

# Otherwise, leave the Python version open for the manager to handle itself
# Drop torchaudio from the initial installation
```

### **Issue 4: Slow Installation/Download**

```bash
# Use uv with --no-cache
uv pip install --no-cache torch torchvision torchaudio

# Use pip with timeout and retry
pip install --default-timeout=100 --retries 10 torch torchvision torchaudio
```

### **WSL2-Specific Issues**

```bash
# 1. Ensure WSL2 is updated
wsl --update

# 2. Install NVIDIA CUDA on WSL from Microsoft Store
# 3. Enable GPU passthrough in .wslconfig:
# [wsl2]
# gpuSupport=true

# 4. Verify in WSL
nvidia-smi
```

---

## **Best Practices & Optimization Tips**

### **Environment Management**

**With uv (Recommended):**

```bash
# Initialize a project
uv init my-dl-project
cd my-dl-project

# Add dependencies
uv add torch --index-url https://download.pytorch.org/whl/cu128
uv add torchvision torchaudio numpy pandas matplotlib jupyter

# Sync dependencies (install everything from pyproject.toml)
uv sync

# Export for sharing
uv pip freeze > requirements.txt

# Create a reproducible lock file
uv lock --refresh
```

**With conda:**

```bash
# Export environment
conda env export > environment.yml

# Create from file
conda env create -f environment.yml

# Update environment
conda env update -f environment.yml --prune
```

### **Performance Optimization**

```python
import torch

# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# Use cudnn benchmark for convolutional networks (if input sizes don't vary)
torch.backends.cudnn.benchmark = True

# Pin memory for faster data transfer to GPU
train_loader = DataLoader(dataset, batch_size=32, pin_memory=True)

# Use non-blocking transfers
data = data.to('cuda', non_blocking=True)
```

### **Memory Optimization**

```python
# Clear cache regularly
torch.cuda.empty_cache()

# Monitor memory usage
print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
print(f"Cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")

# Use in-place operations where possible
x.relu_()  # In-place
# instead of
x = torch.relu(x)
```

### **Monitoring Tools**

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# With process details
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_util --format=csv

# Python monitoring
import nvidia-ml-py3  # pip install nvidia-ml-py3
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
print(f"GPU Util: {util.gpu}%, Memory Util: {util.memory}%")
```

---

## **FAQs**

### **Q: Which CUDA version should I choose?**

**A**: Use the **latest stable CUDA version** that PyTorch supports and that is ≤ your driver version. As of 2024, CUDA 12.8 is recommended for most users.

### **Q: Should I install system CUDA toolkit separately?**

**A**: **No need**. PyTorch bundles its own CUDA toolkit. Only install system CUDA if you need it for other applications.

### **Q: Python 3.13 or older Python?**

**A**: Use **Python 3.10 or 3.11** for best compatibility. Python 3.13 may have limited PyTorch support.

### **Q: uv, pip, or conda?**

**A**:

- **uv**: Best for speed and reliability (seminar recommended)
- **pip**: Simple and universal
- **conda**: Best for complex dependency resolution

### **Q: How to update PyTorch?**

```bash
# uv
uv pip install --upgrade torch torchvision torchaudio

# pip
pip install --upgrade torch torchvision torchaudio

# conda
conda update pytorch torchvision torchaudio pytorch-cuda
```

### **Q: PyTorch detects GPU but training is slow?**

**A**:

1. Check if data transfer is bottleneck: `pin_memory=True` in DataLoader
2. Enable mixed precision: `torch.cuda.amp`
3. Increase batch size (if memory allows)
4. Set `torch.set_float32_matmul_precision('high')`

### **Q: How to share environment with teammates?**

**A**:

- **uv**: Share `pyproject.toml` and `uv.lock`, teammates run `uv sync`
- **pip**: Share `requirements.txt`, teammates run `pip install -r requirements.txt`
- **conda**: Share `environment.yml`, teammates run `conda env create -f environment.yml`

### **Q: AMD GPU support?**

**A**: Yes, via ROCm. Use the ROCm installation commands. Check [AMD ROCm compatibility](https://rocm.docs.amd.com) for your specific GPU.

---

## **Quick Reference Cheat Sheet**

```bash
# 1. Check GPU
nvidia-smi

# 2. Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create environment
uv venv pytorch-env
source pytorch-env/bin/activate

# 4. Install PyTorch (CUDA 12.8)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 5. Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 6. Run benchmark
python benchmark_gpu_cpu.py
```

---

## **Need More Help?**

- **PyTorch Official Docs**: https://pytorch.org/get-started/locally/
- **uv Documentation**: https://docs.astral.sh/uv/
- **NVIDIA CUDA Docs**: https://docs.nvidia.com/cuda/
- **Seminar Forum**: Check your course materials
- **GitHub Issues**: https://github.com/pytorch/pytorch/issues

---

*Last Updated: 01-12-2025*
*GitHub Author: @mchadolias*
=======
  CPU   :  1621.47 ms
  CUDA  :     3.31 ms

Element-wise Operations:
  CPU   :  1190.33 ms
  CUDA  :     5.02 ms

Convolution Operation:
  CPU   :   360.74 ms
  CUDA  :    14.21 ms

============================================================
PERFORMANCE SUMMARY
============================================================
Matrix Multiplication (1000x1000):
  Speedup:  490.1x faster on GPU
  GPU is 490.1x faster than CPU
Element-wise Operations:
  Speedup:  237.2x faster on GPU
  GPU is 237.2x faster than CPU
Convolution Operation:
  Speedup:   25.4x faster on GPU
  GPU is 25.4x faster than CPU

MEMORY BENCHMARK:
  Allocated 95.4 MB on GPU successfully

============================================================
FINAL VERIFICATION
============================================================
✓ Basic GPU computation: SUCCESS
✓ GPU gradient computation: SUCCESS
✓ Final tensor device: cuda:0
✓ GPU memory allocated: 8.7 MB
✓ GPU memory cached: 22.0 MB

🎉 PyTorch GPU setup is working correctly!
============================================================
```

## Additional Recommendations

### Troubleshooting Common Issues

**If CUDA is not available:**

1. Check your NVIDIA driver version: `nvidia-smi`
2. Verify PyTorch CUDA version matches your system capability
3. Try reinstalling with the correct CUDA version

**For WSL Users:**

- Ensure you have WSL 2 with GPU passthrough enabled
- Install NVIDIA CUDA on WSL from the Microsoft Store

### Performance Tips

- Use `torch.set_float32_matmul_precision('high')` for better performance on modern GPUs
- Consider using mixed precision training with `torch.cuda.amp` for larger models
- Monitor GPU usage with `watch -n 1 nvidia-smi` during training

### Virtual Environment Best Practices

```bash
# Using conda (recommended)
conda create -n pytorch-env python=3.10
conda activate pytorch-env

# Using venv
python -m venv pytorch-env
source pytorch-env/bin/activate
```
>>>>>>> upstream/master
