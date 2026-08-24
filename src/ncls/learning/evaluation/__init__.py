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


def evaluate_checkpoint(*args, **kwargs):
    from .evaluator import evaluate_checkpoint as run

    return run(*args, **kwargs)


def evaluate_model(*args, **kwargs):
    from .evaluator import evaluate_model as run

    return run(*args, **kwargs)


__all__ = [
    "QUALITY_SUITE",
    "QUALITY_SUITE_DOCUMENT",
    "QUALITY_SUITE_NAME",
    "QUALITY_SUITE_SHA256",
    "build_quality_report",
    "compare_quality_reports",
    "evaluate_checkpoint",
    "evaluate_model",
    "finalize_quality_report",
    "quality_metric_rows",
    "write_comparison_report",
    "write_quality_report",
]
