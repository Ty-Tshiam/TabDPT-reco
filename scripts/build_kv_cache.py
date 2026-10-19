"""
Precomputes and serializes TabDPT KV cache for context.pt and y.pt.

Used by Triton Server (model_repo/tabdpt/3/model.py) and fast workers
to eliminate context re-encoding latency on startup.
"""

import sys
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

tabdpt_src = PROJECT_ROOT / "src" / "TabDPT-inference" / "src"
if tabdpt_src.exists() and str(tabdpt_src) not in sys.path:
    sys.path.insert(0, str(tabdpt_src))

from src.config import CONTEXT_TENSOR_PATH, KV_CACHE, Y_TENSOR_PATH
from scripts.experiment_context_splits import load_tabdpt_model


def precompute_kv_cache(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print("================================================================================")
    print("  PRECOMPUTING CONTEXT KV CACHE")
    print("================================================================================")
    print(f"  Device:         {device}")
    print(f"  Context tensor: {CONTEXT_TENSOR_PATH}")
    print(f"  Output cache:   {KV_CACHE}")

    model = load_tabdpt_model(device=device)

    x_ctx = torch.load(CONTEXT_TENSOR_PATH, map_location=device).to(dtype=dtype)
    y_ctx = torch.load(Y_TENSOR_PATH, map_location=device).to(dtype=torch.long)
    if x_ctx.ndim == 2:
        x_ctx = x_ctx.unsqueeze(0)
    if y_ctx.ndim == 1:
        y_ctx = y_ctx.unsqueeze(0)

    print(f"  Context shape: {x_ctx.shape}, Labels shape: {y_ctx.shape}")
    t0 = time.time()
    with torch.no_grad():
        kv_cache, n_ctx, stats = model.encode_context(x_ctx, y_ctx, is_cls=True)
    encode_time = time.time() - t0
    print(f"  KV Cache computed in {encode_time:.2f}s (n_ctx={n_ctx})")

    torch.save((kv_cache, n_ctx, stats), str(KV_CACHE))
    cache_mb = Path(KV_CACHE).stat().st_size / (1024 * 1024)
    print(f"  Successfully saved to {KV_CACHE} ({cache_mb:.2f} MB)")
    print("================================================================================\n")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if len(sys.argv) > 1:
        dev = sys.argv[1]
    precompute_kv_cache(device=dev)
