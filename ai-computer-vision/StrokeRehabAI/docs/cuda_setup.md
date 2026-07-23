# CUDA Setup (NVIDIA RTX 3050 Laptop GPU)

## 1. Check your driver

```powershell
nvidia-smi
```

Note the "CUDA Version" shown in the top-right of the output — this is the
*maximum* CUDA toolkit version your driver supports, not necessarily what's
installed.

## 2. Install a matching PyTorch build

Pick the CUDA build that matches (or is older than) your driver's reported
max version, from https://pytorch.org/get-started/locally/. For example, for
CUDA 12.1:

```powershell
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

## 3. Verify GPU visibility from Python

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Or use the project's own check:

```powershell
python main.py check-gpu
```

Expected output includes your GPU name, ~6GB VRAM, and the active CUDA
version (see `utils/gpu_utils.py` / `utils/logger.py:log_gpu_info`).

## 4. RTX 3050 (6GB VRAM) specific notes

- `configs/gpu.yaml` sets `max_batch_size_6gb_vram: 16` as a starting point.
  `utils/gpu_utils.estimate_safe_batch_size()` gives a rough heuristic if you
  change hardware.
- `configs/training.yaml` enables `mixed_precision: true` by default — this
  roughly halves activation memory and is recommended on 6GB cards.
- If you hit `CUDA out of memory`, lower `training.batch_size` in
  `configs/training.yaml` before reducing model size.

## 5. ONNX Runtime GPU (optional, for exported-model inference)

`onnxruntime-gpu` (in requirements.txt) requires a CUDA/cuDNN version
combination compatible with your installed CUDA toolkit — see the
[ONNX Runtime CUDA compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
if `onnxruntime.get_device()` reports `CPU` instead of `GPU`.
