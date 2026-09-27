"""
Standalone Triton-Compatible gRPC Inference Server for TabDPT.

Implements the KServe v2 / Triton gRPC Inference Protocol specification
(GRPCInferenceService) directly on port 8001 using tritonclient's compiled
protobuf descriptors and PyTorch on GPU.
"""

import sys
import time
import os
from pathlib import Path
from concurrent import futures
import numpy as np
import torch
import torch.nn.functional as F
import grpc

# Add TabDPT submodule and project root to sys.path
_repo_root = Path(__file__).resolve().parent.parent
_tabdpt_src = _repo_root / "src" / "TabDPT-inference" / "src"
for p in [_repo_root, _tabdpt_src]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tritonclient.grpc.service_pb2 as pb2
import tritonclient.grpc.service_pb2_grpc as pb2_grpc
from tabdpt.model import TabDPTModel
from src.config import MODEL_CONFIG, KV_CACHE, MODEL_ARTIFACT


class TritonTabDPTService(pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.config = MODEL_CONFIG["settings"]

        print(f"[TritonService] Initializing TabDPTModel on {self.device} ({self.dtype})...")
        self.model = TabDPTModel(**self.config)

        print(f"[TritonService] Loading model weights from {MODEL_ARTIFACT}...")
        state_dict = torch.load(str(MODEL_ARTIFACT), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device, dtype=self.dtype).eval()

        print(f"[TritonService] Loading KV-cache from {KV_CACHE}...")
        self.kv_cache, self.n_ctx, self.stats = torch.load(str(KV_CACHE), map_location=self.device)

        try:
            if torch.cuda.is_available() and hasattr(torch, "compile"):
                print("[TritonService] Compiling predict_with_cache with torch.compile...")
                self.predict_fn = torch.compile(self.model.predict_with_cache)
            else:
                self.predict_fn = self.model.predict_with_cache
        except Exception as e:
            print(f"[TritonService] Compilation note: {e}, using eager mode.")
            self.predict_fn = self.model.predict_with_cache

        print("[TritonService] Model successfully initialized and ready for gRPC inference!")

    def ServerLive(self, request, context):
        return pb2.ServerLiveResponse(live=True)

    def ServerReady(self, request, context):
        return pb2.ServerReadyResponse(ready=True)

    def ModelReady(self, request, context):
        return pb2.ModelReadyResponse(ready=True)

    def ServerMetadata(self, request, context):
        return pb2.ServerMetadataResponse(
            name="tritonserver",
            version="2.72.0",
            extensions=["classification", "sequence_batching"],
        )

    def ModelMetadata(self, request, context):
        response = pb2.ModelMetadataResponse(
            name=request.name or "tabdpt",
            versions=["3"],
            platform="python",
        )
        inp = response.inputs.add()
        inp.name = "QUERY_FEATURES"
        inp.datatype = "FP32"
        inp.shape.extend([1, 1, 128])

        out = response.outputs.add()
        out.name = "PROBABILITIES"
        out.datatype = "FP32"
        out.shape.extend([1, 16])
        return response

    def ModelInfer(self, request, context):
        try:
            # 1. Locate input tensor
            input_meta = None
            input_idx = 0
            for idx, inp in enumerate(request.inputs):
                if inp.name == "QUERY_FEATURES":
                    input_meta = inp
                    input_idx = idx
                    break

            if input_meta is None:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Missing input: QUERY_FEATURES")

            # 2. Extract raw bytes to NumPy array
            shape = list(input_meta.shape)
            raw_bytes = request.raw_input_contents[input_idx]
            x_np = np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape)

            # 3. Convert to GPU tensor
            x_gpu = torch.as_tensor(x_np, device=self.device, dtype=self.dtype)

            # 4. Predict using KV-cache
            with torch.no_grad():
                logits = self.predict_fn(x_gpu, self.kv_cache, self.n_ctx, self.stats)
                if logits.ndim == 3 and logits.shape[0] == 1:
                    logits = logits.squeeze(0)
                logits_16 = logits[..., :16]
                probs = F.softmax(logits_16.float(), dim=-1)

            probs_np = probs.cpu().numpy()  # shape: (1, 16)

            # 5. Pack into Triton ModelInferResponse
            response = pb2.ModelInferResponse(
                model_name=request.model_name or "tabdpt",
                model_version=request.model_version or "3",
                outputs=[
                    pb2.ModelInferResponse.InferOutputTensor(
                        name="PROBABILITIES",
                        datatype="FP32",
                        shape=list(probs_np.shape),
                    )
                ],
                raw_output_contents=[probs_np.tobytes()],
            )
            return response

        except Exception as e:
            print(f"[TritonService] Error during ModelInfer: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))


def serve(port: int = 8001):
    service = TritonTabDPTService()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )
    pb2_grpc.add_GRPCInferenceServiceServicer_to_server(service, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"\n=======================================================")
    print(f"  Triton gRPC Inference Server listening on port {port}")
    print(f"  Ready to serve model 'tabdpt' (v3)")
    print(f"=======================================================\n")
    server.wait_for_termination()


if __name__ == "__main__":
    port = int(os.environ.get("TRITON_GRPC_PORT", "8001"))
    serve(port)
