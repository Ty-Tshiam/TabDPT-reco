"""
Baseline Benchmark Quick Entrypoint.

Provides a fast script to run benchmarks or inspect benchmark models.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_model.run_benchmark import main

if __name__ == "__main__":
    main()