"""
Legacy Stage 1 Module: PySpark to Polars Transition Wrapper.

DEPRECATION NOTICE:
This module previously implemented Stage 1 using PySpark. Stage 1 has been
fully migrated to high-performance native Polars in:
    src.stages.stage1_polars_features

All functions are re-exported here for backwards compatibility.
"""

import warnings

from src.stages.stage1_polars_features import (
    cap_categories_for_tabdpt,
    clean_pipeline,
    engineer_tabdpt_features,
    extract_test_targets,
    load_raw_data,
    run_stage1,
)

warnings.warn(
    "src.stages.stage1_spark_features is deprecated. Stage 1 has been migrated to Polars. "
    "Please import from src.stages.stage1_polars_features instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "cap_categories_for_tabdpt",
    "clean_pipeline",
    "engineer_tabdpt_features",
    "extract_test_targets",
    "load_raw_data",
    "run_stage1",
]


if __name__ == "__main__":
    run_stage1()
