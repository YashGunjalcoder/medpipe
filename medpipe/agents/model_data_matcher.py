"""
Agent 4 — Model-Data Matcher Agent  [Ingestion_Feature_Matcher]

Paper: bridges ingestion and deployment by picking the best-fit model.
  * Tabular: embed every header into a 768-d vector with SapBERT (biomedical
    encoder), compare user headers vs each model's required headers with cosine
    similarity. A model is ELIGIBLE iff every required header matches some
    dataset column with similarity > threshold (empirically 0.6). A greedy
    one-to-one assignment prevents a dataset column matching two required
    headers. Among eligible models the LLM picks the best by description.
    Output: chosen model + filtered dataset containing only required headers
    ("valid data headers").
  * Image: MedGemma uses modality + disease type to select the aligned model.

SUBSTITUTION NOTE: SapBERT downloads from HuggingFace and wants a GPU. The
embedding step is abstracted behind `HeaderEmbedder`. Production uses the
`SapBERTEmbedder`; the offline demo uses `LocalEmbedder` (medical-synonym
normalisation + character n-gram TF-IDF), which yields real cosine similarities
so the threshold/greedy logic is exercised faithfully. Same interface either way.
"""
from __future__ import annotations
import numpy as np

from .base import Agent, PipelineContext
from ..model_database import get_tabular_models, get_image_models

DEFAULT_THRESHOLD = 0.6

# small curated biomedical synonym map so the OFFLINE embedder can bridge
# obvious clinical equivalences that SapBERT would capture natively.
_SYNONYMS = {
    "sex": "gender", "gender": "gender",
    "dob": "age", "age": "age",
    "bmi": "body mass index",
    "sppb": "short physical performance battery",
    "tug": "timed up and go",
    "gds": "geriatric depression scale",
    "adl": "activities of daily living",
    "ecog": "ecog performance status",
    "pal": "palliative", "palliative": "palliative",
}


def _normalize(h: str) -> str:
    h = str(h).lower().replace("_", " ").replace("-", " ").strip()
    toks = [_SYNONYMS.get(t, t) for t in h.split()]
    return " ".join(toks)


class HeaderEmbedder:
    def fit_transform(self, headers: list[str]) -> np.ndarray:
        raise NotImplementedError


class LocalEmbedder(HeaderEmbedder):
    """Offline: char n-gram TF-IDF over synonym-normalised headers."""
    def fit_transform(self, headers):
        from sklearn.feature_extraction.text import TfidfVectorizer
        norm = [_normalize(h) for h in headers]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        return vec.fit_transform(norm).toarray()


class SapBERTEmbedder(HeaderEmbedder):
    """Production: 768-d SapBERT embeddings (cambridgeltl/SapBERT-...)."""
    def __init__(self, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        from transformers import AutoTokenizer, AutoModel
        import torch
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def fit_transform(self, headers):
        import torch
        with torch.no_grad():
            enc = self.tok(list(headers), padding=True, truncation=True,
                           return_tensors="pt", max_length=25)
            out = self.model(**enc)[0][:, 0]  # CLS token, 768-d
        return out.cpu().numpy()


def _cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


class ModelDataMatcherAgent(Agent):
    name = "ModelDataMatcherAgent"

    def __init__(self, embedder: HeaderEmbedder | None = None,
                 threshold: float = DEFAULT_THRESHOLD):
        self.embedder = embedder or LocalEmbedder()
        self.threshold = threshold

    def _run(self, ctx: PipelineContext) -> str:
        if ctx.data_kind == "tabular":
            return self._match_tabular(ctx)
        return self._match_image(ctx)

    def _match_tabular(self, ctx: PipelineContext) -> str:
        user_headers = ctx.headers
        models = get_tabular_models()

        eligible = {}
        per_model = {}
        for mname, meta in models.items():
            required = meta["headers"]
            # exclude the target/output header from matching requirements
            required = [h for h in required if h != meta.get("output")]
            all_headers = user_headers + required
            emb = self.embedder.fit_transform(all_headers)
            U = emb[:len(user_headers)]
            R = emb[len(user_headers):]

            # similarity matrix required x user
            sim = np.array([[_cosine(r, u) for u in U] for r in R])

            # greedy one-to-one: each required header takes its best free column
            assigned_cols = set()
            matches = {}
            ok = True
            order = np.argsort(-sim.max(axis=1))  # match easiest first
            for ri in order:
                best_j, best_s = -1, -1.0
                for j in range(len(user_headers)):
                    if j in assigned_cols:
                        continue
                    if sim[ri, j] > best_s:
                        best_s, best_j = sim[ri, j], j
                if best_j >= 0 and best_s > self.threshold:
                    assigned_cols.add(best_j)
                    matches[required[ri]] = (user_headers[best_j], round(best_s, 3))
                else:
                    ok = False
            per_model[mname] = {"matched": matches, "n_required": len(required),
                                "n_matched": len(matches), "eligible": ok}
            if ok:
                eligible[mname] = matches

        # LLM-picks-best step: here choose eligible model that maps the MOST
        # user columns (proxy for the paper's description-based LLM choice).
        if eligible:
            chosen = max(eligible, key=lambda m: per_model[m]["n_matched"])
            valid = [col for (col, _s) in eligible[chosen].values()]
            ctx.selected_model = chosen
            ctx.valid_headers = valid
            ctx.target = ctx.forced_target or models[chosen]["output"]
        else:
            ctx.selected_model = None
            ctx.valid_headers = []
            ctx.target = ctx.forced_target

        ctx.match_report = {"threshold": self.threshold, "per_model": per_model,
                            "chosen": ctx.selected_model}
        return (f"chosen={ctx.selected_model}, valid_headers={len(ctx.valid_headers)}"
                if ctx.selected_model else "no eligible model (all headers below threshold)")

    def _match_image(self, ctx: PipelineContext) -> str:
        models = get_image_models()
        cands = {m: meta for m, meta in models.items()
                 if meta["modality"] == ctx.image_modality}
        if not cands:
            ctx.selected_model = None
            return f"no image model for modality '{ctx.image_modality}'"
        # MedGemma would pick by disease/caption; proxy = first modality match
        chosen = sorted(cands)[0]
        ctx.selected_model = chosen
        ctx.match_report = {"candidates": list(cands), "chosen": chosen,
                            "modality": ctx.image_modality, "disease": ctx.image_disease}
        return f"chosen image model={chosen} for modality='{ctx.image_modality}'"
