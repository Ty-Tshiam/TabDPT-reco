"""
End-to-end integration test for Triton gRPC Inference.

Validates:
1. Server liveness and readiness via gRPC.
2. Model readiness for 'tabdpt'.
3. Sending QUERY_FEATURES tensor [1, 1, 128] with FP32 datatype.
4. Receiving PROBABILITIES tensor [1, 16] with valid probability distribution.
5. Recommendation ranking against the 16 target product classes.
"""

import sys
from pathlib import Path
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import tritonclient.grpc as grpcclient
from src.config import SELECTED_15_TARGETS, OTHER_9_PRODUCTS, TARGET_TO_INDEX, INDEX_TO_TARGET


def test_triton_grpc_inference():
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # 1. Health checks
    assert client.is_server_live(), "Triton Server is not live"
    assert client.is_server_ready(), "Triton Server is not ready"
    assert client.is_model_ready("tabdpt"), "Model 'tabdpt' is not ready"

    # 2. Prepare query [Batch=1, Queries=1, Features=128]
    np.random.seed(42)
    query_features = np.random.randn(1, 1, 128).astype(np.float32)

    inputs = [grpcclient.InferInput("QUERY_FEATURES", [1, 1, 128], "FP32")]
    inputs[0].set_data_from_numpy(query_features)

    outputs = [grpcclient.InferRequestedOutput("PROBABILITIES")]

    # 3. Run gRPC inference
    response = client.infer(model_name="tabdpt", inputs=inputs, outputs=outputs)
    probs = response.as_numpy("PROBABILITIES")

    # 4. Assertions
    assert probs is not None, "Inference response returned None"
    assert probs.shape == (1, 16), f"Expected shape (1, 16), got {probs.shape}"
    assert np.isclose(probs.sum(), 1.0, atol=1e-3), f"Probabilities do not sum to 1: {probs.sum()}"
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities out of [0, 1] range"

    # 5. Class ranking
    probs_1d = probs.squeeze()
    sorted_indices = np.argsort(-probs_1d)
    ranked_products = [INDEX_TO_TARGET[i] for i in sorted_indices]

    print("\n--- Triton gRPC Test Results ---")
    print(f"Top-1 Recommended Product: {ranked_products[0]} (p={probs_1d[sorted_indices[0]]:.4f})")
    print("Top-7 Product Recommendations:")
    for rank, idx in enumerate(sorted_indices[:7], 1):
        print(f"  {rank}. {INDEX_TO_TARGET[idx]:<25} (prob = {probs_1d[idx]:.4f})")


if __name__ == "__main__":
    test_triton_grpc_inference()
    print("\nAll Triton gRPC tests passed successfully!")
