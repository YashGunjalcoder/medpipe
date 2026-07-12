"""
Agents 5 & 6 — Preprocessing Recommender + Implementor
                [Preprocessing_Recommender] / [Preprocessing_Implementor]

Recommender (paper): for tabular data, extract per-column metadata (name,
dtype, #null, #unique, min, max) and infer a column TYPE with a rule-based
heuristic:
  * Binary      -> exactly 2 unique values
  * Categorical -> low #unique with short string lengths
  * Numerical   -> numeric dtype AND #unique > 0.8 * n_rows
  * Textual     -> anything else
The inferred type drives the recommended preprocessing steps. Above a size
threshold (50 MB) user selection is disabled and steps are auto-selected.
For image data, preprocessing is model-coupled (DETR-style co-trained
transforms), so the recommender defers to the selected model.

Implementor (paper): executes the recommended steps on the anonymized,
feature-matched data. Imputation/scaling are fit on train only (no leakage).
"""
from __future__ import annotations
import numpy as np

from .base import Agent, PipelineContext

SIZE_THRESHOLD_MB = 50


def infer_column_type(series, n_rows: int) -> str:
    import pandas as pd
    nunique = series.nunique(dropna=True)
    if nunique == 2:
        return "Binary"
    if pd.api.types.is_numeric_dtype(series) and nunique > 0.8 * n_rows:
        return "Numerical"
    if series.dtype == object:
        avg_len = series.dropna().astype(str).str.len().mean() or 0
        if nunique <= max(20, 0.05 * n_rows) and avg_len <= 25:
            return "Categorical"
        return "Textual"
    # numeric but not high-cardinality -> treat as categorical codes
    if pd.api.types.is_numeric_dtype(series) and nunique <= 20:
        return "Categorical"
    return "Numerical"


_STEPS = {
    "Binary":      ["impute_mode", "as_int"],
    "Categorical": ["impute_mode", "one_hot_encode"],
    "Numerical":   ["impute_median", "standard_scale"],
    "Textual":     ["drop_or_embed"],
}


class PreprocessingRecommenderAgent(Agent):
    name = "PreprocessingRecommenderAgent"

    def _run(self, ctx: PipelineContext) -> str:
        if ctx.data_kind == "image":
            ctx.preprocessing_plan = {"mode": "model_coupled",
                                      "note": "DETR co-trained transforms owned by the image model."}
            return "image preprocessing delegated to model-specific pipeline"

        import pandas as pd
        df: pd.DataFrame = ctx.dataframe
        # keep only valid headers (from matcher) + target
        cols = [c for c in ctx.valid_headers if c in df.columns]
        if ctx.target and ctx.target in df.columns and ctx.target not in cols:
            cols = cols + [ctx.target]
        df = df[cols].copy() if cols else df.copy()
        ctx.dataframe = df

        n = len(df)
        col_types, plan, meta = {}, {}, {}
        for c in df.columns:
            if c == ctx.target:
                continue
            t = infer_column_type(df[c], n)
            col_types[c] = t
            plan[c] = _STEPS[t]
            s = df[c]
            meta[c] = {"dtype": str(s.dtype), "n_null": int(s.isna().sum()),
                       "n_unique": int(s.nunique(dropna=True))}
        ctx.column_types = col_types
        ctx.preprocessing_plan = {"mode": "tabular", "steps": plan, "metadata": meta}
        return f"typed {len(col_types)} columns: " + \
               ", ".join(f"{k}={v}" for k, v in list(col_types.items())[:6]) + \
               ("..." if len(col_types) > 6 else "")


class PreprocessingImplementorAgent(Agent):
    name = "PreprocessingImplementorAgent"

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def _run(self, ctx: PipelineContext) -> str:
        if ctx.data_kind == "image":
            return "image preprocessing handled inside model inference (model-coupled)"

        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        df: pd.DataFrame = ctx.dataframe.copy()
        target = ctx.target
        if target not in df.columns:
            raise ValueError(f"target '{target}' not in dataframe")

        # target framing:
        #   binary     -> collapse ordinal severity (0 vs >=1)
        #   multiclass  -> keep distinct integer classes
        #   auto        -> binary if the raw target has >2 levels but is ordinal
        #                  numeric (severity scale), else use as-is
        raw = df[target]
        mode = ctx.target_mode
        if mode == "auto":
            if raw.dtype != object and raw.nunique(dropna=True) > 2:
                mode = "binary"   # ordinal severity scale -> present/absent
            else:
                mode = "binary" if raw.nunique(dropna=True) <= 2 else "multiclass"

        if mode == "binary":
            if raw.dtype == object:
                y = pd.factorize(raw)[0]
                y = (y == y.max()).astype(int) if len(set(y)) == 2 else y
            else:
                # 0 = absent/no-data, >=1 = present
                y = (raw.fillna(0) >= 1).astype(int)
        else:  # multiclass
            y = pd.factorize(raw)[0] if raw.dtype == object else raw.astype(int)

        ctx.n_classes = int(len(np.unique(y)))
        X = df.drop(columns=[target])

        # apply per-column steps
        num_cols, encoded_frames = [], []
        for c in X.columns:
            steps = ctx.preprocessing_plan["steps"].get(c, [])
            s = X[c]
            if "one_hot_encode" in steps:
                s = s.fillna(s.mode(dropna=True).iloc[0] if not s.mode().empty else "NA")
                dummies = pd.get_dummies(s.astype(str), prefix=c, dtype=float)
                encoded_frames.append(dummies)
            elif "impute_median" in steps or "standard_scale" in steps:
                s = pd.to_numeric(s, errors="coerce")
                s = s.fillna(s.median())
                num_cols.append(c)
                encoded_frames.append(s.to_frame(c))
            elif "as_int" in steps or "impute_mode" in steps:
                s = pd.to_numeric(s, errors="coerce")
                s = s.fillna(s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else 0)
                num_cols.append(c)
                encoded_frames.append(s.to_frame(c))
            else:  # textual -> drop
                continue

        Xp = pd.concat(encoded_frames, axis=1) if encoded_frames else X.select_dtypes("number")

        X_tr, X_te, y_tr, y_te = train_test_split(
            Xp, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None)

        # scale numeric columns — FIT ON TRAIN ONLY (no leakage; paper's concern)
        scale_cols = [c for c in num_cols if c in X_tr.columns]
        if scale_cols:
            sc = StandardScaler()
            X_tr = X_tr.copy(); X_te = X_te.copy()
            X_tr[scale_cols] = sc.fit_transform(X_tr[scale_cols])
            X_te[scale_cols] = sc.transform(X_te[scale_cols])

        ctx.X_train, ctx.X_test, ctx.y_train, ctx.y_test = X_tr, X_te, y_tr, y_te
        return f"train={X_tr.shape}, test={X_te.shape}, features={X_tr.shape[1]}"
