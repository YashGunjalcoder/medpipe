"""
Model Database (paper Figures 2 & 4).

A curated registry of candidate models. Each tabular model advertises the
`headers` (feature columns) it requires plus its prediction `output`. Each
image model advertises its `modality` and a `caption` describing its task.

The Model-Data Matcher Agent searches this registry to find the best-fit model
for a user's uploaded data via SapBERT-embedding cosine similarity.

MODEL_01 mirrors the real HOPE palliative-care anxiety schema (Hofmann et al.
2017). MODEL_02 (hope) and MODEL_03 (GSTRIDE falls) are alternative models used
to demonstrate that the matcher correctly *rejects* mismatched schemas.
"""

MODEL_DATABASE = {
    "table": {
        # Palliative-care anxiety prediction — matches the HOPE dataset.
        "MODEL_01": {
            "modality": "anxiety prediction (palliative care)",
            "headers": [
                "age", "gender", "ECOG", "living_situation", "pal_care_service",
                "brain_metastases", "pain", "nausea", "vomiting", "dyspnea",
                "constipation", "weakness", "loss_appetite", "tiredness",
                "assistance_adl", "antidepressants", "sedatives_anxiolytics",
                "corticosteroids", "feeling_depressed", "tension",
                "disorientation_confusion",
            ],
            "output": "anxiety",
            "task": "classification",
        },
        # Hope-level prediction — overlapping but distinct required schema.
        "MODEL_02": {
            "modality": "hope prediction (palliative care)",
            "headers": [
                "age", "gender", "antidepressants", "vomiting", "pain",
                "assistance_adl", "loss_appetite", "feeling_depressed",
            ],
            "output": "hope_level",
            "task": "classification",
        },
        # Geriatric fall prediction (GSTRIDE) — deliberately mismatched schema.
        "MODEL_03": {
            "modality": "fall prediction (geriatrics / GSTRIDE)",
            "headers": [
                "age", "sex", "bmi", "stride_length", "stride_time",
                "walking_speed", "foot_clearance", "foot_angle", "sppb", "tug",
                "grip_strength", "gds",
            ],
            "output": "faller",
            "task": "classification",
        },
    },
    "image": {
        "MODEL_06": {
            "modality": "colon colonoscopy scan",
            "caption": "Detects and classifies hyperplastic vs. adenomatous polyps in colonoscopy images.",
            "task": "detection",
        },
        "MODEL_07": {
            "modality": "breast histopathology scan",
            "caption": "Classify benign vs. malignant findings in breast histopathology slides.",
            "task": "classification",
        },
        "MODEL_09": {
            "modality": "colon colonoscopy scan",
            "caption": "Detects and localizes lesions in colonoscopy images.",
            "task": "detection",
        },
    },
}


def get_tabular_models():
    return MODEL_DATABASE["table"]


def get_image_models():
    return MODEL_DATABASE["image"]
