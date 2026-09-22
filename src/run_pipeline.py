"""
TabDPT Product Recommendation Pipeline CLI Runner.

Provides a unified command-line interface to execute data preparation and feature
engineering stages individually or end-to-end:

Usage:
    # Run the complete pipeline (Stage 1 -> Stage 2 -> Stage 3)
    python src/run_pipeline.py --stage all

    # Run individual stages
    python src/run_pipeline.py --stage 1    # PySpark cleaning & feature engineering
    python src/run_pipeline.py --stage 2    # Categorical encoding & global normalization
    python src/run_pipeline.py --stage 3    # Stratified sampling & PyTorch tensors
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_directories_exist
from src.stages.stage1_spark_features import run_stage1
from src.stages.stage2_encode_normalize import run_stage2
from src.stages.stage3_tensor_builder import run_stage3


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TabDPT Santander Product Recommendation Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1: PySpark Data Cleaning & Feature Engineering (Raw CSV -> Parquet)
  2: Categorical Frequency Encoding & Global Normalization (Features -> Normalized Parquet)
  3: Stratified Subsampling & PyTorch Tensor Export (Normalized Parquet -> context.pt, y.pt)
  all: Run Stages 1, 2, and 3 sequentially
        """
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Pipeline stage to execute (default: all)"
    )
    return parser.parse_args()


def main():
    """Main execution entrypoint."""
    args = parse_args()
    ensure_directories_exist()

    print("=" * 70)
    print(f"TabDPT Pipeline Runner - Target Stage: {args.stage.upper()}")
    print("=" * 70)
    start_time = time.time()

    if args.stage in ("1", "all"):
        print("\n>>> Launching Stage 1: PySpark Data Cleaning & Feature Engineering...")
        t0 = time.time()
        run_stage1()
        print(f">>> Stage 1 Finished in {time.time() - t0:.1f}s")

    if args.stage in ("2", "all"):
        print("\n>>> Launching Stage 2: Categorical Frequency Encoding & Global Normalization...")
        t0 = time.time()
        run_stage2()
        print(f">>> Stage 2 Finished in {time.time() - t0:.1f}s")

    if args.stage in ("3", "all"):
        print("\n>>> Launching Stage 3: Stratified Sampling & PyTorch Tensor Export...")
        t0 = time.time()
        run_stage3()
        print(f">>> Stage 3 Finished in {time.time() - t0:.1f}s")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Pipeline finished successfully in {elapsed:.1f}s!")
    print("=" * 70)


if __name__ == "__main__":
    main()
