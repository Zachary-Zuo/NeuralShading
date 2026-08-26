from .comparison import compare_quality_reports, write_comparison_report
from .quality import (
    QUALITY_SUITE,
    QUALITY_SUITE_DOCUMENT,
    QUALITY_SUITE_NAME,
    QUALITY_SUITE_SHA256,
    build_quality_report,
    finalize_quality_report,
    quality_metric_rows,
    write_quality_report,
)
from .unified_selection import (
    build_unified_selection_manifest,
    build_unified_selection_from_artifacts,
    compare_unified_cells,
    load_unified_selection_protocol,
    paired_state_difference,
    unified_selection_protocol_sha256,
    write_unified_selection_manifest,
)


def evaluate_checkpoint(*args, **kwargs):
    from .evaluator import evaluate_checkpoint as run

    return run(*args, **kwargs)


def evaluate_model(*args, **kwargs):
    from .evaluator import evaluate_model as run

    return run(*args, **kwargs)


def benchmark_checkpoint(*args, **kwargs):
    from .benchmark import benchmark_checkpoint as run

    return run(*args, **kwargs)


def run_unified_offline_cook(*args, **kwargs):
    from .offline_cook import run_unified_offline_cook as run

    return run(*args, **kwargs)


def run_unified_checkpoint_parity(*args, **kwargs):
    from .unified_parity import run_unified_checkpoint_parity as run

    return run(*args, **kwargs)


def run_convergence_audit(*args, **kwargs):
    from .convergence import run_convergence_audit as run

    return run(*args, **kwargs)


def run_sampler_convergence_audit(*args, **kwargs):
    from .convergence import run_sampler_convergence_audit as run

    return run(*args, **kwargs)


__all__ = [
    "QUALITY_SUITE",
    "QUALITY_SUITE_DOCUMENT",
    "QUALITY_SUITE_NAME",
    "QUALITY_SUITE_SHA256",
    "build_quality_report",
    "build_unified_selection_manifest",
    "build_unified_selection_from_artifacts",
    "benchmark_checkpoint",
    "compare_quality_reports",
    "compare_unified_cells",
    "evaluate_checkpoint",
    "evaluate_model",
    "finalize_quality_report",
    "load_unified_selection_protocol",
    "paired_state_difference",
    "quality_metric_rows",
    "run_unified_offline_cook",
    "run_unified_checkpoint_parity",
    "run_convergence_audit",
    "run_sampler_convergence_audit",
    "write_comparison_report",
    "write_quality_report",
    "unified_selection_protocol_sha256",
    "write_unified_selection_manifest",
]
