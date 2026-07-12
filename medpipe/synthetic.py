"""
Generate a synthetic tabular dataset matching the HOPE palliative-care anxiety
schema (Hofmann et al. 2017). Used only by `medpipe demo` so the pipeline can be
showcased with no external file.

IMPORTANT: this data is SYNTHETIC. Metrics from `demo` are illustrative only,
NOT clinical results. Run `medpipe run <real_file>` for real numbers.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def generate(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def sev(p_present=0.6):  # 0..3 severity scale like the real symptoms
        base = rng.integers(0, 4, n)
        mask = rng.random(n) > p_present
        base[mask] = 0
        return base

    age = rng.normal(69, 12, n).clip(19, 105).round(1)
    gender = rng.integers(1, 3, n)                 # 1=female, 2=male
    ecog = rng.integers(0, 5, n)
    living_situation = rng.integers(0, 4, n)
    pal_care_service = rng.integers(0, 4, n)
    brain_metastases = rng.integers(0, 2, n)
    pain = sev(); nausea = sev(); vomiting = sev(); dyspnea = sev()
    constipation = sev(); weakness = sev(); loss_appetite = sev()
    tiredness = sev(); assistance_adl = rng.integers(0, 4, n)
    antidepressants = rng.integers(0, 2, n)
    sedatives_anxiolytics = rng.integers(0, 2, n)
    corticosteroids = rng.integers(0, 2, n)
    feeling_depressed = sev(); tension = sev()
    disorientation_confusion = sev()

    logit = (
        0.85 * tension + 0.70 * feeling_depressed + 0.45 * loss_appetite
        + 0.40 * vomiting + 0.25 * ecog - 0.45 * antidepressants
        + 0.35 * brain_metastases + 0.03 * (age - 69)
        + rng.normal(0, 0.6, n) - 2.4
    )
    p = 1 / (1 + np.exp(-logit))
    anxiety = np.where(rng.random(n) < p, rng.integers(1, 4, n), 0)

    return pd.DataFrame({
        "age": age, "gender": gender, "ECOG": ecog,
        "living_situation": living_situation, "pal_care_service": pal_care_service,
        "brain_metastases": brain_metastases, "pain": pain, "nausea": nausea,
        "vomiting": vomiting, "dyspnea": dyspnea, "constipation": constipation,
        "weakness": weakness, "loss_appetite": loss_appetite, "tiredness": tiredness,
        "assistance_adl": assistance_adl, "antidepressants": antidepressants,
        "sedatives_anxiolytics": sedatives_anxiolytics, "corticosteroids": corticosteroids,
        "feeling_depressed": feeling_depressed, "tension": tension,
        "disorientation_confusion": disorientation_confusion,
        "anxiety": anxiety,
    })


if __name__ == "__main__":
    df = generate()
    df.to_csv("anxiety_synthetic.csv", index=False)
    print(df.shape)
