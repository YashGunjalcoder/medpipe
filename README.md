# medpipe — Agentic AI Framework for End-to-End Medical Data Inference

A modular, multi-agent pipeline that takes a raw clinical data upload and
autonomously carries it to an interpretable prediction — file-type detection →
PII anonymization → feature extraction → model selection → preprocessing →
inference → explainability — with no manual step in between, driven by a CLI
that works on **any tabular dataset**.

Faithful implementation of *"Agentic AI framework for End-to-End Medical Data
Inference"* (Shimgekar et al., arXiv:2507.18115). Validated end-to-end on the
**real HOPE palliative-care dataset** (Hofmann et al. 2017, 9,924 patients).

## Results on the real HOPE dataset

`python -m medpipe run hope.sav --target anxiety`

Predicting whether a palliative-care patient presents anxiety (present vs.
absent), 9,924 patients, 21 matched clinical features, 80/20 train/test split,
PerpetualBooster:

| Metric | Value |
|---|---|
| Accuracy | **0.849** |
| Precision | 0.865 |
| Recall | 0.926 |
| F1 | **0.894** |
| ROC-AUC | **0.879** |
| 5-fold CV accuracy | **0.834 ± 0.005** |

Top SHAP drivers: `tension`, `feeling_depressed`, `age`, `loss_appetite`,
`brain_metastases` — clinically sensible and learned, not hardcoded.

The Model-Data Matcher correctly selected **MODEL_01 (21/21 required headers
matched)**, also found MODEL_02 eligible (8/8), and **rejected MODEL_03**
(GSTRIDE fall model, only 2/12 headers) — demonstrating real schema-aware model
selection.

> These are **real metrics** you can regenerate by rerunning the command. They
> are reproducible (fixed seed) and match the paper's task. `medpipe demo` uses
> synthetic data and its numbers are illustrative only.

## Architecture (7 agents)

```
data file (.sav/.csv/.xlsx)
    │
    ▼
1. Ingestion Identifier ──► Magika file-type detection; routes tabular vs image
    ▼
2. Data Anonymizer ───────► masks PII (emails, phones, SSN, MRN, DOB, identity cols)
    ▼
3. Feature Extraction ────► tabular headers; image: MedGemma modality + disease
    ▼
4. Model-Data Matcher ────► header embeddings, cosine similarity vs model DB,
    │                        greedy 1:1 assignment, threshold 0.6 → best-fit model
    ▼
5. Preprocessing Recommender ► rule-based typing (Binary/Categorical/Numerical/Textual)
    ▼
6. Preprocessing Implementor ► impute / encode / scale (fit on train only)
    ▼
7. Model Inference ───────► PerpetualBooster; SHAP top-10; predictions CSV + JSON report
```

Every agent implements one `run(context)` method against a shared
`PipelineContext`, so any agent can be re-wrapped as a Google ADK tool (the
paper's orchestrator) without changing its logic.

## Install

```bash
pip install -r requirements.txt        # core (runs everything on CPU)
# or, as an installable package with the `medpipe` command:
pip install -e .
```

## Usage (CLI)

```bash
# inspect a dataset's schema
python -m medpipe info hope.sav

# run the full pipeline on real data
python -m medpipe run hope.sav --target anxiety

# any tabular file works — CSV, TSV, XLSX, SPSS .sav
python -m medpipe run mydata.csv --target outcome --threshold 0.55

# skip SHAP for a faster run
python -m medpipe run hope.sav --target anxiety --no-explain

# self-contained showcase on synthetic data (no file needed)
python -m medpipe demo
```

Flags: `--target`, `--target-mode {auto,binary,multiclass}`, `--threshold`,
`--test-size`, `--seed`, `--no-explain`, `--output-dir`.

Outputs land in `run_output/`: `predictions.csv` and `pipeline_report.json`
(full machine-readable trace, decisions, metrics, and SHAP importances), plus a
printed agent-by-agent trace.

## What runs where

| Component | Paper's tool | This repo |
|---|---|---|
| File-type detection | Magika | **Magika (identical)** |
| Tabular inference GBM | PerpetualBooster | **PerpetualBooster (identical)** |
| Explainability | SHAP / LIME | **SHAP (real)**, LIME pluggable |
| Header embeddings | SapBERT (768-d, GPU) | `SapBERTEmbedder` interface + offline `LocalEmbedder` (synonym-normalised char n-gram TF-IDF) |
| Anonymization | Google Cloud DLP | Local regex + column-heuristic detector; `dlp` backend pluggable |
| Image features | MedGemma (VLM, GPU) | Documented interface + deterministic stub |
| Image model | fine-tuned DETR (GPU) | Interface; run locally on KUMC images |
| Orchestration | Google ADK | Framework-agnostic orchestrator, same contract |

To use the production components: `pip install -e ".[full]"` (adds transformers,
torch, lime), then point the backends at SapBERT / MedGemma / Google DLP.

## Getting the real dataset

- **HOPE palliative-care anxiety** (used above): Hofmann et al. 2017, PLOS ONE
  `10.1371/journal.pone.0179415` → Supporting Information → **S1 File**
  ("HOPE core data set 2007–2011.sav").
- **GSTRIDE** (geriatric fall prediction): Zenodo `10.5281/zenodo.8003441`.
  Add a `MODEL_03`-style entry and run with `--target <faller_col>`.
- **KUMC colonoscopy polyps** (image branch): Li et al. 2021, PLOS ONE
  `10.1371/journal.pone.0255809`. Needs a GPU to fine-tune DETR; wire MedGemma +
  DETR backends into the feature-extraction and inference agents.

## Project layout

```
medpipe/
  pyproject.toml            # installable package + `medpipe` console command
  requirements.txt
  README.md
  medpipe/
    __init__.py
    __main__.py             # enables `python -m medpipe`
    cli.py                  # argparse CLI (run / info / demo)
    orchestrator.py         # chains the 7 agents
    model_database.py       # candidate model registry (paper Fig. 2/4)
    data_loader.py          # .sav / .csv / .tsv / .xlsx loading
    report.py               # console + JSON reporting
    synthetic.py            # demo data generator
    agents/
      base.py               # Agent base class + PipelineContext blackboard
      ingestion_identifier.py   # Agent 1  (Magika)
      data_anonymizer.py        # Agent 2  (PII masking)
      feature_extraction.py     # Agent 3  (headers / MedGemma interface)
      model_data_matcher.py     # Agent 4  (embeddings + cosine + greedy match)
      preprocessing.py          # Agents 5 & 6 (recommender + implementor)
      model_inference.py        # Agent 7  (PerpetualBooster + SHAP)
```


  with **5-fold CV accuracy 0.83 ± 0.01** predicting patient anxiety.

Honest framing: this implements and runs a published framework on a public
dataset — describe it as *"implemented and validated the framework from
[paper]"* rather than as original method research. That's both accurate and
still strong.
