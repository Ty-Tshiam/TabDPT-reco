import os
import sys
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Ensure TabDPT library submodule is on sys.path
tabdpt_src = Path(__file__).resolve().parent / "TabDPT-inference" / "src"
if tabdpt_src.exists() and str(tabdpt_src) not in sys.path:
    sys.path.insert(0, str(tabdpt_src))

from tabdpt.model import TabDPTModel

try:
    from src.config import (
        CONTEXT_TENSOR_PATH,
        KV_CACHE,
        MODEL_ARTIFACT,
        MODEL_CONFIG,
        Y_TENSOR_PATH,
    )
except ImportError:
    from config import (
        CONTEXT_TENSOR_PATH,
        KV_CACHE,
        MODEL_ARTIFACT,
        MODEL_CONFIG,
        Y_TENSOR_PATH,
    )

config = MODEL_CONFIG["settings"]
repo_id = "Layer6/TabDPT"
model = TabDPTModel(**config)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32


def return_tensors():
    context = torch.load(CONTEXT_TENSOR_PATH, map_location=device)
    y_train = torch.load(Y_TENSOR_PATH, map_location=device)

    context = context.to(device, dtype=dtype)
    y_train = y_train.to(device, dtype=torch.long)

    if context.ndim == 2:
        context = context.unsqueeze(0)
    if y_train.ndim == 1:
        y_train = y_train.unsqueeze(0)

    return y_train, context


def save_model():
    try:
        weights_path = hf_hub_download(
            repo_id=repo_id, filename="tabdpt1_3.safetensors", local_files_only=False
        )
        weight_file = "tabdpt1_3.safetensors"
    except Exception as e:
        print("No 1.3 using 1.2")
        weights_path = hf_hub_download(
            repo_id=repo_id, filename="tabdpt1_2.safetensors", local_files_only=False
        )
        weight_file = "tabdpt1_2.safetensors"

    print(f"[Model] Downloading HuggingFace weights {weight_file} from {weights_path}")
    print(f"[Model] Loading Model on {device} with {dtype}")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.to(device, dtype=dtype).eval()

    rebuild = "--force" in sys.argv
    if os.path.exists(KV_CACHE) and not rebuild:
        try:
            cached = torch.load(str(KV_CACHE), map_location="cpu")
            if not isinstance(cached, tuple) or len(cached) != 3 or not isinstance(cached[2], dict):
                print("[Model] Existing cache format is outdated, re-encoding...")
                rebuild = True
        except Exception:
            rebuild = True

    if os.path.exists(KV_CACHE) and not rebuild:
        print("[Model] Already Cached")
    else:
        print("[Model] Encoding context (will cache to disk)...")
        y_ctx, x_ctx = return_tensors()
        with torch.no_grad():
            try:
                if torch.cuda.is_available() and hasattr(torch, "compile"):
                    encode_fn = torch.compile(model.encode_context)
                else:
                    encode_fn = model.encode_context
            except Exception:
                encode_fn = model.encode_context
            kv_cache, n_ctx, stats = encode_fn(x_ctx, y_ctx, is_cls=True)
        torch.save((kv_cache, n_ctx, stats), str(KV_CACHE))
        print(f"[Model] Saved KV cache to {KV_CACHE}")

    os.makedirs(os.path.dirname(str(MODEL_ARTIFACT)), exist_ok=True)
    torch.save(model.state_dict(), str(MODEL_ARTIFACT))
    print(f"[Model] Saved Model artifact to {MODEL_ARTIFACT}")


if __name__ == "__main__":
    save_model()
    print("Success!")