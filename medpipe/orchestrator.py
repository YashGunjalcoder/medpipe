"""
Orchestrator — chains the 7 agents end-to-end (paper Figure 1).

  Ingestion Identifier -> Data Anonymizer -> Feature Extraction
  -> Model-Data Matcher -> Preprocessing Recommender
  -> Preprocessing Implementor -> Model Inference

Each agent implements the same run(context) contract against a shared
PipelineContext, so any agent can be re-wrapped as a Google ADK tool (the
paper's orchestrator) without changing its logic.
"""
from __future__ import annotations

from .agents.base import PipelineContext
from .agents.ingestion_identifier import IngestionIdentifierAgent
from .agents.data_anonymizer import DataAnonymizerAgent
from .agents.feature_extraction import FeatureExtractionAgent
from .agents.model_data_matcher import ModelDataMatcherAgent
from .agents.preprocessing import (PreprocessingRecommenderAgent,
                                   PreprocessingImplementorAgent)
from .agents.model_inference import ModelInferenceAgent


class Orchestrator:
    def __init__(self, output_dir: str = ".", threshold: float = 0.6,
                 explain: bool = True, test_size: float = 0.2,
                 random_state: int = 42):
        self.explain = explain
        self.agents = [
            IngestionIdentifierAgent(),
            DataAnonymizerAgent(backend="local"),
            FeatureExtractionAgent(),
            ModelDataMatcherAgent(threshold=threshold),
            PreprocessingRecommenderAgent(),
            PreprocessingImplementorAgent(test_size=test_size,
                                          random_state=random_state),
            ModelInferenceAgent(output_dir=output_dir, explain=explain),
        ]

    def run(self, input_path: str, dataframe=None,
            force_target: str | None = None,
            target_mode: str = "auto") -> PipelineContext:
        ctx = PipelineContext(input_path=input_path)
        ctx.dataframe = dataframe
        ctx.forced_target = force_target
        ctx.target_mode = target_mode
        for agent in self.agents:
            agent.run(ctx)
            if agent.name == "ModelDataMatcherAgent" and ctx.selected_model is None \
                    and ctx.data_kind == "tabular":
                ctx.trace.append({"agent": "Orchestrator", "status": "halted",
                                  "detail": "no eligible model; pipeline stopped"})
                break
        return ctx
