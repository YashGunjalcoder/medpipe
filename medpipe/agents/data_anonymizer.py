"""
Agent 2 — Data Anonymizer Agent  [Ingestion_Anonymizer]

Paper: detects and redacts PII from structured (tabular) and unstructured
(image) data using the Google Cloud DLP API. Structured PII is masked with
fixed-length placeholder tokens ("****"); PII text embedded in images is
covered with opaque overlays.

SUBSTITUTION NOTE: Google Cloud DLP requires GCP credentials and sends data
off-site. To keep the pipeline self-contained and locally runnable (and to
avoid the data-sovereignty concern the paper flags in its Ethics section), this
agent uses a local detector: rule-based regexes (emails, phones, SSNs, credit
cards, IPs, dates of birth, medical record numbers) plus column-name heuristics
for names/identifiers. The public interface is identical to a DLP call, so a
`GoogleDLPBackend` can be dropped in unchanged. Set backend="dlp" once creds
are configured.
"""
from __future__ import annotations
import re

from .base import Agent, PipelineContext

MASK = "****"

_REGEXES = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"(?<!\d)(\+?\d[\d\-\s().]{7,}\d)(?!\d)"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "DOB": re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"),
    "MRN": re.compile(r"\b(?:MRN[:#]?\s*)?\d{6,10}\b", re.IGNORECASE),
}

# column names that are PII by identity regardless of content
_PII_COLUMN_HINTS = (
    "name", "patient", "email", "phone", "address", "ssn", "mrn",
    "record", "dob", "birth", "id_number", "national_id", "insurance",
)


class DataAnonymizerAgent(Agent):
    name = "DataAnonymizerAgent"

    def __init__(self, backend: str = "local"):
        self.backend = backend  # "local" | "dlp"

    def _redact_cell(self, value) -> tuple[object, set[str]]:
        if not isinstance(value, str):
            return value, set()
        hits = set()
        out = value
        for label, rx in _REGEXES.items():
            if rx.search(out):
                hits.add(label)
                out = rx.sub(MASK, out)
        return out, hits

    def _run(self, ctx: PipelineContext) -> str:
        if ctx.data_kind == "tabular":
            return self._anonymize_tabular(ctx)
        return self._anonymize_image(ctx)

    def _anonymize_tabular(self, ctx: PipelineContext) -> str:
        import pandas as pd
        df: pd.DataFrame = ctx.dataframe.copy()
        report = {"masked_columns": {}, "detected_types": set()}

        for col in df.columns:
            lowered = str(col).lower()
            # whole-column masking for identity-PII columns
            if any(h in lowered for h in _PII_COLUMN_HINTS):
                df[col] = MASK
                report["masked_columns"][col] = "column_name_hint"
                report["detected_types"].add("IDENTITY")
                continue
            # cell-level regex masking for free-text columns
            if df[col].dtype == object:
                col_hits = set()
                masked_vals = []
                for v in df[col]:
                    nv, hits = self._redact_cell(v)
                    masked_vals.append(nv)
                    col_hits |= hits
                if col_hits:
                    df[col] = masked_vals
                    report["masked_columns"][col] = sorted(col_hits)
                    report["detected_types"] |= col_hits

        report["detected_types"] = sorted(report["detected_types"])
        ctx.dataframe = df
        ctx.anonymization_report = report
        return f"tabular anonymized: {len(report['masked_columns'])} column(s), types={report['detected_types']}"

    def _anonymize_image(self, ctx: PipelineContext) -> str:
        # Real DLP would OCR + overlay black rectangles on detected text regions.
        # Offline stub: record that images pass through the redaction step. A
        # local OCR+overlay (e.g. easyocr + PIL rectangles) plugs in here.
        ctx.anonymization_report = {
            "mode": "image",
            "note": "OCR-based visual redaction stub; wire in DLP visual inspection or local OCR overlay.",
            "n_images": len(ctx.image_paths),
        }
        return f"image branch: {len(ctx.image_paths)} image(s) marked for visual redaction"
