"""
Agent 7 — Model Inference Agent  [Model_Inferencer]

Paper: applies the selected model to processed data and produces interpretable
outputs.
  * Small tabular data -> PerpetualBooster (a gradient boosting machine).
    Large data -> a custom deep architecture.
  * Explainability via SHAP and LIME; report top-10 most influential features.
  * Image data -> fine-tuned DETR; bounding boxes + attention maps + CSV of
    coordinates/labels.
Outputs are written to a structured CSV.

This implementation uses the *real* PerpetualBooster (same as the paper) for
tabular inference and *real* SHAP for explanations. The image/DETR branch is a
documented interface — training DETR needs the KUMC images + a GPU, so run that
half locally with the provided links.
"""
from __future__ import annotations
import numpy as np

from .base import Agent, PipelineContext


class ModelInferenceAgent(Agent):
    name = "ModelInferenceAgent"

    def __init__(self, output_dir: str = ".", large_threshold_rows: int = 100_000,
                 explain: bool = True):
        self.output_dir = output_dir
        self.large_threshold = large_threshold_rows
        self.explain = explain

    def _run(self, ctx: PipelineContext) -> str:
        if ctx.data_kind == "image":
            ctx.metrics = {"note": "DETR inference runs locally on KUMC images (GPU)."}
            return "image inference delegated to DETR (see README for local run)"
        return self._infer_tabular(ctx)

    def _infer_tabular(self, ctx: PipelineContext) -> str:
        import os
        import pandas as pd
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix)
        from perpetual import PerpetualBooster

        X_tr, X_te = ctx.X_train, ctx.X_test
        y_tr, y_te = ctx.y_train, ctx.y_test

        n_classes = ctx.n_classes or len(np.unique(y_tr))
        model = PerpetualBooster(objective="LogLoss" if n_classes == 2 else "LogLoss")
        model.fit(X_tr, y_tr)

        # predictions
        proba = None
        try:
            proba = model.predict_proba(X_te)
        except Exception:
            pass
        if proba is not None and getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
            y_prob = proba[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            raw = model.predict(X_te)
            raw = np.asarray(raw).ravel()
            if set(np.unique(raw)) <= {0, 1}:
                y_pred, y_prob = raw.astype(int), raw.astype(float)
            else:  # scores/logits -> sigmoid
                y_prob = 1 / (1 + np.exp(-raw))
                y_pred = (y_prob >= 0.5).astype(int)

        avg = "binary" if n_classes == 2 else "macro"
        metrics = {
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "n_features": int(X_tr.shape[1]),
            "accuracy": round(accuracy_score(y_te, y_pred), 4),
            "precision": round(precision_score(y_te, y_pred, average=avg, zero_division=0), 4),
            "recall": round(recall_score(y_te, y_pred, average=avg, zero_division=0), 4),
            "f1": round(f1_score(y_te, y_pred, average=avg, zero_division=0), 4),
        }
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_te, y_prob), 4)
        except Exception:
            metrics["roc_auc"] = None
        metrics["confusion_matrix"] = confusion_matrix(y_te, y_pred).tolist()

        # 5-fold CV accuracy on full processed data for a stability estimate
        # (manual loop — PerpetualBooster isn't a drop-in sklearn estimator)
        try:
            from sklearn.model_selection import StratifiedKFold
            X_all = pd.concat([X_tr, X_te]).reset_index(drop=True)
            y_all = np.concatenate([np.asarray(y_tr), np.asarray(y_te)])
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            fold_acc = []
            for tr_idx, te_idx in skf.split(X_all, y_all):
                fm = PerpetualBooster()
                fm.fit(X_all.iloc[tr_idx], y_all[tr_idx])
                fp = fm.predict_proba(X_all.iloc[te_idx])
                if getattr(fp, "ndim", 1) == 2 and fp.shape[1] >= 2:
                    fpred = (fp[:, 1] >= 0.5).astype(int)
                else:
                    fr = np.asarray(fm.predict(X_all.iloc[te_idx])).ravel()
                    fpred = (fr >= 0.5).astype(int) if set(np.unique(fr)) - {0, 1} else fr.astype(int)
                fold_acc.append(accuracy_score(y_all[te_idx], fpred))
            metrics["cv5_accuracy_mean"] = round(float(np.mean(fold_acc)), 4)
            metrics["cv5_accuracy_std"] = round(float(np.std(fold_acc)), 4)
        except Exception as e:
            metrics["cv5_accuracy_mean"] = None
            metrics["cv_note"] = str(e)

        ctx.metrics = metrics

        # write predictions CSV
        os.makedirs(self.output_dir, exist_ok=True)
        pred_path = os.path.join(self.output_dir, "predictions.csv")
        out = X_te.copy()
        out["y_true"] = np.asarray(y_te)
        out["y_pred"] = y_pred
        out["y_score"] = np.round(y_prob, 4)
        out.to_csv(pred_path, index=False)
        ctx.predictions_path = pred_path

        # SHAP explanations — top-10 influential features
        if self.explain:
            self._explain(ctx, model, X_tr, X_te)
        else:
            ctx.explanations = {"method": "SHAP", "note": "skipped (--no-explain)"}

        return (f"acc={metrics['accuracy']} f1={metrics['f1']} "
                f"auc={metrics['roc_auc']} cv5={metrics.get('cv5_accuracy_mean')}")

    def _explain(self, ctx, model, X_tr, X_te):
        try:
            import shap
            bg = X_tr.sample(min(100, len(X_tr)), random_state=0)
            expl = shap.Explainer(model.predict, bg)
            sv = expl(X_te.iloc[:min(100, len(X_te))])
            vals = np.abs(sv.values)
            if vals.ndim == 3:
                vals = vals.mean(axis=2)
            importance = vals.mean(axis=0)
            order = np.argsort(-importance)[:10]
            ctx.explanations = {
                "method": "SHAP",
                "top10": [(str(X_te.columns[i]), round(float(importance[i]), 5)) for i in order],
            }
        except Exception as e:
            ctx.explanations = {"method": "SHAP", "error": str(e)}
