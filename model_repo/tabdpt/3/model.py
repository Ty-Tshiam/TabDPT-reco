import triton_python_backend_utils as pb_utils  # Note: 'utils', not 'untils'
import torch
import torch.nn.functional as F

import os
import sys
from pathlib import Path

# Ensure repo root and TabDPT submodule are on sys.path
_this_dir = Path(__file__).resolve()
for candidate_dir in [
    _this_dir.parents[3],                                   # repo root from model_repo/tabdpt/3/model.py
    _this_dir.parents[3] / "src" / "TabDPT-inference" / "src",
    Path("/workspace"),
    Path("/workspace/TabDPT-reco"),
    Path("/workspace/TabDPT-reco/src/TabDPT-inference/src"),
]:
    if candidate_dir.exists() and str(candidate_dir) not in sys.path:
        sys.path.insert(0, str(candidate_dir))

# 1. Import your model architecture definition
from tabdpt.model import TabDPTModel

# 2. Import your configuration paths
try:
    from src.config import MODEL_CONFIG, KV_CACHE, MODEL_ARTIFACT
except ImportError:
    from config import MODEL_CONFIG, KV_CACHE, MODEL_ARTIFACT


class TritonPythonModel:
    def initialize(self, args):
        """Called ONCE when Triton loads the model version."""
        self.config = MODEL_CONFIG["settings"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Instantiate architecture and load weights
        print(f"[Inference] Initializing TabDPTModel...")
        self.model = TabDPTModel(**self.config)

        print(f"[Inference] Loading Model weights from {MODEL_ARTIFACT}")
        state_dict = torch.load(str(MODEL_ARTIFACT), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device, dtype=self.dtype).eval()

        # Load precomputed KV cache (Context and y are baked into this!)
        print(f"[Inference] Loading KV Cache from {KV_CACHE}")
        self.kv_cache, self.n_ctx, self.stats = torch.load(str(KV_CACHE), map_location=self.device)

        # Optimize inference method with torch.compile if supported
        try:
            if torch.cuda.is_available() and hasattr(torch, "compile"):
                self.predict_fn = torch.compile(self.model.predict_with_cache)
            else:
                self.predict_fn = self.model.predict_with_cache
        except Exception:
            self.predict_fn = self.model.predict_with_cache

    def execute(self, requests):
        """
        Called on every inference cycle. 
        'requests' contains 1 or more incoming queries batched by Triton.
        """
        responses = []
        batch_tensors = []

        # 1. Extract inputs from all requests in this dynamic batch
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "QUERY_FEATURES")
            # Convert NumPy -> PyTorch tensor on GPU with target dtype
            tensor_gpu = torch.as_tensor(
                in_tensor.as_numpy(), 
                device=self.device, 
                dtype=self.dtype
            )
            batch_tensors.append(tensor_gpu)

        # Concatenate into shape [B, 1, 128]
        x_qry_batch = torch.cat(batch_tensors, dim=0)

        # 2. Run your custom model inference with KV cache
        with torch.no_grad():
            logits = self.predict_fn(x_qry_batch, self.kv_cache, self.n_ctx, self.stats)
            if logits.ndim == 3 and logits.shape[0] == 1:
                logits = logits.squeeze(0)
            logits_16 = logits[..., :16]
            probs = F.softmax(logits_16.float(), dim=-1)  # Shape: [B, 16]

        # 3. Convert results back to CPU NumPy for Triton serialization
        probs_np = probs.to(torch.float32).cpu().numpy()

        # 4. Pack each prediction into an InferenceResponse matching its request
        for i in range(len(requests)):
            # Slice row i -> shape [1, 16]
            out_tensor = pb_utils.Tensor("PROBABILITIES", probs_np[i : i + 1])
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        """Called when Triton unloads the model."""
        del self.model
        del self.kv_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()