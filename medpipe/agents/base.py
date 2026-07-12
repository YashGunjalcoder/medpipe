"""
Base classes for the agentic pipeline.

The paper orchestrates agents with Google's Agent Development Kit (ADK). ADK
requires Google Cloud credentials, so here we use a lightweight, framework-
agnostic orchestrator with the SAME contract: each agent has a single
`run(context)` entrypoint, reads what it needs from a shared PipelineContext,
writes its outputs back, and logs a structured trace. Swapping in ADK later is
a matter of wrapping each agent's `run` as an ADK tool.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineContext:
    """Shared state threaded through every agent (the 'blackboard')."""
    # inputs
    input_path: str | None = None
    forced_target: str | None = None   # user-specified target column (CLI --target)
    # ingestion identifier
    file_types: dict[str, str] = field(default_factory=dict)
    data_kind: str | None = None          # "tabular" | "image"
    # anonymizer
    dataframe: Any = None                 # pandas DataFrame for tabular
    anonymization_report: dict = field(default_factory=dict)
    image_paths: list[str] = field(default_factory=list)
    # feature extraction
    headers: list[str] = field(default_factory=list)
    image_modality: str | None = None
    image_disease: str | None = None
    # model-data matcher
    selected_model: str | None = None
    valid_headers: list[str] = field(default_factory=list)
    match_report: dict = field(default_factory=dict)
    # preprocessing
    column_types: dict[str, str] = field(default_factory=dict)
    preprocessing_plan: dict = field(default_factory=dict)
    X_train: Any = None
    X_test: Any = None
    y_train: Any = None
    y_test: Any = None
    target: str | None = None
    target_mode: str = "auto"          # "auto" | "binary" | "multiclass"
    n_classes: int | None = None
    # inference
    metrics: dict = field(default_factory=dict)
    predictions_path: str | None = None
    explanations: dict = field(default_factory=dict)
    # trace
    trace: list[dict] = field(default_factory=list)


class Agent:
    """Every agent subclasses this and implements `_run`."""
    name = "Agent"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        status = "ok"
        detail = ""
        try:
            detail = self._run(ctx) or ""
        except Exception as e:  # agents fail gracefully and record the failure
            status = "error"
            detail = f"{type(e).__name__}: {e}"
            raise
        finally:
            dt = time.perf_counter() - t0
            ctx.trace.append(
                {"agent": self.name, "status": status, "seconds": round(dt, 4), "detail": detail}
            )
        return ctx

    def _run(self, ctx: PipelineContext) -> str:
        raise NotImplementedError
