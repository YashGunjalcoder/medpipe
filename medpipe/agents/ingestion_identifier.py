"""
Agent 1 — Ingestion Identifier Agent  [Ingestion_Classifier]

Paper: classifies uploaded files (CSV, XLSX, JSON, ZIP) using Magika, a deep
learning-based MIME-type detector. For ZIP archives it recursively unpacks
in-memory and classifies each inner file. Produces a structured summary of the
unique file types and routes the pipeline to the tabular or image branch.

This implementation uses the *real* Magika library (same as the paper).
"""
from __future__ import annotations
import io
import os
import zipfile

from .base import Agent, PipelineContext

_TABULAR_LABELS = {"csv", "tsv", "xls", "xlsx", "json", "jsonl", "txt", "parquet",
                   "sav", "dta", "sas7bdat"}
_IMAGE_LABELS = {"jpeg", "jpg", "png", "bmp", "tiff", "gif", "webp"}


class IngestionIdentifierAgent(Agent):
    name = "IngestionIdentifierAgent"

    def __init__(self):
        from magika import Magika
        self._m = Magika()

    def _label(self, data: bytes) -> str:
        res = self._m.identify_bytes(data)
        # magika API differs slightly across versions; handle both
        try:
            return res.output.ct_label
        except AttributeError:
            return res.dl.ct_label

    def _run(self, ctx: PipelineContext) -> str:
        path = ctx.input_path
        found: dict[str, str] = {}

        def classify(name: str, data: bytes):
            if zipfile.is_zipfile(io.BytesIO(data)):
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for inner in zf.namelist():
                        if inner.endswith("/"):
                            continue
                        classify(inner, zf.read(inner))
                return
            found[name] = self._label(data)

        with open(path, "rb") as f:
            classify(os.path.basename(path), f.read())

        ctx.file_types = found
        labels = set(found.values())
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        # statistical/binary tabular formats Magika may label 'unknown'
        if ext in ("sav", "dta", "sas7bdat"):
            ctx.data_kind = "tabular"
        elif labels & _IMAGE_LABELS:
            ctx.data_kind = "image"
        elif labels & _TABULAR_LABELS:
            ctx.data_kind = "tabular"
        else:
            ctx.data_kind = "image" if ext in _IMAGE_LABELS else "tabular"

        return f"detected file_types={found} -> data_kind={ctx.data_kind}"
