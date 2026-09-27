# Running Mosaic on Kaggle / Colab

Prerequisite: `src/decision_engine/training/device_utils.py` gives every script
`device=None` auto-detection (`cuda` if available, else `cpu`) -- nothing below
requires editing code, but it didn't work before this was added (the codebase never
called `.to(device)` anywhere; it would have silently trained on CPU in a GPU notebook).

Both platforms ship PyTorch **pre-installed with a matching CUDA build** -- don't
`pip install torch` again, that's how you end up with a CPU-only wheel shadowing the
GPU one. Only install what's missing.

## Kaggle (recommended -- see PLAN.md/README for why: 30 free GPU-hrs/week, published
quota, no credit card, same setup Laya's own team used)

1. kaggle.com -> **Create** -> **New Notebook**.
2. Right panel -> **Accelerator** -> **GPU T4 x2** (or P100). -> **Internet** -> **On**
   (needed for `git clone`, the Phase 1 loaders' live API calls, and the first NeoBERT
   download).
3. First cell:
   ```bash
   %cd /kaggle/working
   !git clone <your-repo-url> mosaic
   %cd mosaic
   !pip install -q transformers accelerate einops safetensors xformers
   !pip install -q -e .
   ```
4. Confirm the GPU is actually visible before running anything:
   ```python
   import torch
   print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
   ```
5. Run any script unchanged -- device auto-detection picks up the GPU:
   ```bash
   !python scripts/train_moe_eval.py
   ```
6. Kaggle notebooks are ephemeral between sessions. To keep a checkpoint or the
   generated `data/` files, either **Save Version** (persists `/kaggle/working`) or
   write results back to a Kaggle Dataset from within the notebook.

## Google Colab (good for quick one-off runs; no published quota, session can be
reclaimed without warning -- don't rely on it for a multi-hour run you can't restart)

1. colab.research.google.com -> **New notebook**.
2. **Runtime** -> **Change runtime type** -> **T4 GPU**.
3. First cell:
   ```bash
   !git clone <your-repo-url> mosaic
   %cd mosaic
   !pip install -q transformers accelerate einops safetensors xformers
   !pip install -q -e .
   ```
4. Same GPU check and run commands as Kaggle, above.
5. Colab's filesystem is wiped when the runtime recycles. Mount Drive first if you
   want checkpoints or generated data to survive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   then point `scripts/train_moe.py`'s output paths (or a checkpoint-saving call you
   add) at `/content/drive/MyDrive/...`.

## Once you're actually on a GPU, don't keep the CPU-smoke-test settings

Every script defaults to CPU-smoke-test sizing (`batch_size=2`, `n_steps=10-20`,
truncated sequences) because that's what this local machine could handle. A T4 has
16GB VRAM -- for a ~250-300M-parameter model that's enormous headroom. Increase
`batch_size` (16-32+), `n_steps` (hundreds, not tens), and `max_state_len` (toward
NeoBERT's real 4,096-token context) once you're actually training for real, not just
proving the pipeline works. The current defaults would badly under-use a real GPU.

## If xformers warns about a version mismatch

The generic `pip install xformers` may not exactly match Kaggle/Colab's pinned
torch+CUDA build, producing the same "can't load C++/CUDA extensions" warning seen on
this CPU machine. It's not fatal -- NeoBERT's modeling file only needs the package
*importable*, and falls back to plain attention -- but if you want xformers' actual
optimized kernels for speed, match its version to the notebook's own torch build
(check `torch.__version__` first, then install the corresponding xformers wheel per
https://github.com/facebookresearch/xformers#installing-xformers).
