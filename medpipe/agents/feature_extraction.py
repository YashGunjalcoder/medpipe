"""
Agent 3 — Feature Extraction Agent  [Ingestion_Selector]

Paper: modality-specific semantic "header" extraction.
  * Tabular: column names are used directly as feature descriptors ("headers").
  * Image: a multi-stage MedGemma (medical vision-language model) pipeline.
      Stage 1 (Prompt 1): given the set of known modalities, pick the single
                          modality that best matches the image.
      Stage 2 (Prompt 2): given that modality, classify the most likely
                          disease type. Confidence thresholds/decision rules
                          allow terminating or re-sampling a new random image.

SUBSTITUTION NOTE: MedGemma runs on GPU and downloads from HuggingFace. The
image path here is a documented interface (`MedGemmaBackend`) with a
deterministic offline stub so the pipeline stays runnable end-to-end. Point
`backend` at a real MedGemma endpoint for production.
"""
from __future__ import annotations

from .base import Agent, PipelineContext
from ..model_database import get_image_models


class FeatureExtractionAgent(Agent):
    name = "FeatureExtractionAgent"

    def __init__(self, medgemma_backend=None):
        self._medgemma = medgemma_backend  # callable(prompt, image) -> str, or None

    def _run(self, ctx: PipelineContext) -> str:
        if ctx.data_kind == "tabular":
            ctx.headers = [str(c) for c in ctx.dataframe.columns]
            return f"extracted {len(ctx.headers)} tabular headers"
        return self._extract_image(ctx)

    # ---- image branch (multi-stage MedGemma) --------------------------------
    def _extract_image(self, ctx: PipelineContext) -> str:
        modalities = sorted({m["modality"] for m in get_image_models().values()})
        sample = ctx.image_paths[0] if ctx.image_paths else None

        if self._medgemma is not None and sample is not None:
            # Stage 1: modality
            p1 = (f"Given these modalities {modalities}, return only the one "
                  f"modality that best matches the image content.")
            modality = self._medgemma(p1, sample).strip()
            # Stage 2: disease type conditioned on modality + caption
            caption = next((m["caption"] for m in get_image_models().values()
                            if m["modality"] == modality), "")
            p2 = (f"Given the modality '{modality}' and its context '{caption}', "
                  f"identify and return the most relevant disease type in the image.")
            disease = self._medgemma(p2, sample).strip()
        else:
            # deterministic offline stub for demonstration
            modality = "colon colonoscopy scan"
            disease = "Polyp"

        ctx.image_modality = modality
        ctx.image_disease = disease
        return f"image headers -> modality='{modality}', disease='{disease}'"
