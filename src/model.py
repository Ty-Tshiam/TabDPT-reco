import torch 
import os
from tabdpt.model import TabDPTModel
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

try:
    from src.config import (
        MODEL_CONFIG,
        MODEL_ARTIFACT,
        KV_CACHE,
        CONTEXT_TENSOR_PATH,
        Y_TENSOR_PATH
    )
except ImportError:
    from config import (
        MODEL_CONFIG,
        MODEL_ARTIFACT,
        KV_CACHE,
        CONTEXT_TENSOR_PATH,
        Y_TENSOR_PATH
    )

config = MODEL_CONFIG["settings"]
repo_id = "Layer6/TabDPT"
model = TabDPTModel(**config)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

def return_tensors():

    context = torch.load(CONTEXT_TENSOR_PATH)
    y_train = torch.load(Y_TENSOR_PATH)

    context = context.to(device, dtype = dtype)
    y_train = y_train.to(device, dtype = torch.long)

    if device == "cpu":
        context = context.numpy()
        y_train = y_train.numpy()
        return y_train, context
        
    else:
        context = context.unsqueeze(0)
        y_train = y_train.unsqueeze(0)
        return y_train, context

def save_model():

    try:
        weights_path = hf_hub_download(
            repo_id=repo_id, filename="tabdpt1_3.safetensors", local_files_only = False
            )
        weight_file = "tabdpt1_3.safetensors"
    except Exception as e:
        print("No 1.3 using 1.2")
        weights_path = hf_hub_download(
            repo_id=repo_id, filename="tabdpt1_2.safetensors", local_files_only = False
            )
        weight_file = "tabdpt1_2.safetensors"
        
    print(f'[Model] Downloading HuggingFace weights {weight_file} from {weights_path}')

    print(f'[Model] Loading Model on {device} with {dtype}')
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.to(device, dtype = dtype).eval()

    if os.path.exists(KV_CACHE):
        print("[Model] Already Cached")
    else:
        print("[Model] Encoding context (first run — will cache for next time)")
        y_ctx, x_ctx = return_tensors()
        with torch.no_grad():
            model.encode_context = torch.compile(model.encode_context)
            kv_cache, n_ctx, stats = model.encode_context(x_ctx, y_ctx, is_cls=True)
        torch.save((kv_cache, n_ctx, stats), KV_CACHE)
        
    torch.save(model.state_dict(), str(MODEL_ARTIFACT))
'''
    with torch.no_grad():
        model.predict_with_cache = torch.compile(model.predict_with_cache)
        logits = model.predict_with_cache(x_qry, kv_cache, n_ctx, stats)
'''
    #model = torch.compile(model, mode = "reduce-overhead")
    

if __name__ == "__main__":
    save_model()
    print("Success!")