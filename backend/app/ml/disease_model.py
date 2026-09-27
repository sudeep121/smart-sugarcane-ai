"""Sugarcane disease classification.

Two clearly separated modes:

* ``trained_model`` - a Keras model saved at ``models/sugarcane_disease_model.keras``
  is loaded and used. TensorFlow is imported lazily so it is never a hard
  dependency of the web API.
* ``demo_heuristic`` - no trained model is present, so a transparent colour and
  texture heuristic produces a *demonstration* estimate. Every response built
  from this mode is labelled DEMO and carries the measured features that drove
  it, so nothing is hidden behind a fake accuracy number.

The heuristic is NOT a validated diagnostic tool and the code never claims it is.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import settings
from app.ml import model_runtime
from app.ml import image_features as feats

logger = logging.getLogger(__name__)

MODEL_STEMS = ("sugarcane_disease_model",)
META_FILENAME = "disease_model_meta.json"

#: Classes the demo heuristic can produce. A trained model may define its own
#: list - whatever is in the model metadata wins at runtime.
DEFAULT_CLASS_NAMES = [
    "healthy",
    "red_rot",
    "rust",
    "smut",
    "mosaic",
    "leaf_scald",
    "yellow_leaf",
]

#: Below this the answer is reported as "unknown / low confidence" instead of a guess.
CONFIDENCE_FLOOR = 0.42

_lock = threading.Lock()
_model = None
_meta: dict[str, Any] = {}
_load_attempted = False
_load_error: str | None = None


# --------------------------------------------------------------------- model IO
def find_model_file() -> Path | None:
    """Preferring .onnx, which serves in a fifth of the memory Keras needs."""
    return model_runtime.find_model_file(settings.models_dir, MODEL_STEMS)


def _read_class_names() -> list[str]:
    meta_path = settings.models_dir / META_FILENAME
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            names = data.get("class_names")
            if isinstance(names, list) and names:
                return [str(n) for n in names]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", meta_path, exc)

    fallback = settings.project_root / "ml" / "disease_detection" / "class_names.json"
    if fallback.exists():
        try:
            data = json.loads(fallback.read_text(encoding="utf-8"))
            names = data.get("class_names", data if isinstance(data, list) else None)
            if isinstance(names, list) and names:
                return [str(n) for n in names]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", fallback, exc)
    return list(DEFAULT_CLASS_NAMES)


def _load_model():
    global _model, _meta, _load_attempted, _load_error

    with _lock:
        if _load_attempted:
            return _model
        _load_attempted = True

        if settings.MODEL_MODE == "demo":
            _load_error = "MODEL_MODE=demo - trained image models are disabled by configuration."
            return None

        path = find_model_file()
        if path is None:
            _load_error = (
                f"No trained disease model found in {settings.models_dir}. "
                "Expected sugarcane_disease_model.onnx or .keras"
            )
            return None

        try:
            _model = model_runtime.load(path)
        except ImportError:
            _load_error = (
                "A model file exists but no runtime can load it. "
                "Install onnxruntime, or TensorFlow for a .keras file."
            )
            logger.warning(_load_error)
            return None
        except Exception as exc:  # noqa: BLE001
            _load_error = f"Could not load {path.name}: {exc}"
            logger.warning(_load_error)
            return None

        meta_path = settings.models_dir / META_FILENAME
        if meta_path.exists():
            try:
                _meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                _meta = {}
        _meta.setdefault("class_names", _read_class_names())

        try:
            shape = _model.input_shape
            _meta.setdefault("input_size", [int(shape[1]), int(shape[2])])
        except Exception:  # noqa: BLE001
            _meta.setdefault("input_size", [224, 224])

        _load_error = None
        logger.info("Loaded trained disease model from %s", path)
        return _model


def reload_model() -> None:
    global _model, _meta, _load_attempted, _load_error
    with _lock:
        _model = None
        _meta = {}
        _load_attempted = False
        _load_error = None


def is_available() -> bool:
    return _load_model() is not None


def status() -> dict[str, Any]:
    model = _load_model()
    path = find_model_file()
    if model is None:
        return {
            "trained_model_available": False,
            "model_path": str(path) if path else str(settings.models_dir / f"{MODEL_STEMS[0]}.onnx"),
            "model_source": "demo_heuristic",
            "mode": settings.MODEL_MODE,
            "class_names": _read_class_names(),
            "notes": [
                _load_error or "No trained model present.",
                "Disease analysis is running in DEMO mode: a transparent colour and texture "
                "heuristic, not a trained neural network.",
                "Results in this mode are illustrative only and must not be used as a diagnosis.",
                "To train a real model see docs/DATASET_GUIDE.md and run ml/disease_detection/train.py",
            ],
        }
    return {
        "trained_model_available": True,
        "model_path": str(path),
        "model_source": "trained_model",
        "mode": settings.MODEL_MODE,
        "class_names": list(_meta.get("class_names", DEFAULT_CLASS_NAMES)),
        "notes": [
            f"Input size: {_meta.get('input_size')}",
            f"Trained at: {_meta.get('trained_at', 'unknown')}",
            f"Reported validation accuracy: {_meta.get('val_accuracy', 'unknown')}",
            "Accuracy depends entirely on the dataset used for training. Always verify "
            "important decisions with a qualified agricultural expert.",
        ],
    }


# ----------------------------------------------------------------- demo scoring
def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    shifted = (scores - scores.max()) / max(temperature, 1e-6)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def _heuristic_scores(f: feats.ImageFeatures) -> dict[str, float]:
    """Weighted colour / texture evidence per class.

    Each weight encodes an ordinary visual observation - for example "rust shows
    many small orange-brown pustules, so brown coverage combined with a highly
    fragmented brown mask is evidence for rust, while the same brown coverage in
    a few large patches points to red rot instead". These are hand-set
    demonstration weights, not learned parameters.

    Colour coverage is measured RELATIVE TO VISIBLE LEAF TISSUE rather than to
    the whole frame. Background, sky and shadow otherwise dominate the ratios and
    every photo of a leaf against dark soil scores the same way.
    """
    tissue = max(f.tissue_ratio, 0.05)
    yellow = f.yellow_ratio / tissue
    brown = f.brown_ratio / tissue
    bleached = f.bleached_ratio / tissue
    # Dark pixels are scored against tissue plus dark, so a dark background does
    # not read as necrosis.
    dark = f.dark_ratio / max(f.tissue_ratio + f.dark_ratio, 0.05)

    def saturating(value: float, full_at: float) -> float:
        """Evidence that reaches its maximum at `full_at` coverage.

        Symptom coverage is small in absolute terms - 3 % of the leaf covered in
        rust pustules is a real infection - so raw ratios have to be amplified
        before they can compete with a healthy leaf's ~100 % green.
        """
        return min(max(value, 0.0) / full_at, 1.0)

    brown_e = saturating(brown, 0.06)
    yellow_e = saturating(yellow, 0.15)
    bleached_e = saturating(bleached, 0.04)
    dark_e = saturating(dark, 0.25)
    affected_e = saturating(brown + yellow + bleached + dark, 0.10)

    # Boundary-to-area ratio of the brown mask: near 0.03 for a few large red-rot
    # lesions, above 0.30 for scattered rust pustules. This is what separates the
    # two, since both show similar colours.
    fragmentation = saturating(f.brown_fragmentation, 0.25)

    # Hue spread inside green tissue is the mosaic signal, with a dead zone: a
    # healthy leaf photographed in mixed light already varies by a few degrees,
    # so nothing counts as mottling until it clears that.
    hue_var = min(max(f.green_hue_std - 5.0, 0.0) / 9.0, 1.0)

    streak = f.vertical_streak_score
    low_saturation = max(0.0, 0.35 - f.mean_saturation) / 0.35

    return {
        "healthy": 3.6 * (1.0 - affected_e) - 1.4 * hue_var,
        # Large contiguous discoloration - fragmentation counts against it.
        "red_rot": 3.4 * brown_e - 2.6 * fragmentation + 0.8 * yellow_e - 1.2 * bleached_e,
        # Same colour as red rot, but broken into many small pustules.
        "rust": 1.8 * brown_e + 2.6 * fragmentation - 1.4 * dark_e - 1.0 * bleached_e,
        "smut": 3.8 * dark_e + 1.0 * low_saturation - 1.4 * yellow_e,
        "mosaic": 3.4 * hue_var + 0.8 * (1.0 - affected_e) - 2.0 * brown_e - 1.8 * dark_e - 1.4 * bleached_e,
        "leaf_scald": 4.0 * bleached_e + 1.4 * streak - 1.6 * brown_e - 1.2 * dark_e,
        "yellow_leaf": 3.6 * yellow_e + 1.0 * streak - 1.8 * brown_e - 1.6 * bleached_e - 1.2 * dark_e,
    }


#: Evidence spread below which the heuristic is treated as genuinely undecided.
#: Dividing by this instead of by the raw spread means weak evidence produces a
#: flat distribution (and therefore an honest "unknown"), while strong evidence
#: produces a decisive answer. Plain min-max normalisation would always push the
#: top class to 1.0 and hide the difference.
MIN_EVIDENCE_SPREAD = 1.0
DEMO_TEMPERATURE = 0.22


def _demo_probabilities(scores: dict[str, float]) -> tuple[list[str], np.ndarray]:
    class_names = list(scores)
    raw = np.array([scores[name] for name in class_names], dtype=np.float64)
    spread = float(raw.max() - raw.min())
    normalised = (raw - raw.min()) / max(spread, MIN_EVIDENCE_SPREAD)
    return class_names, _softmax(normalised, temperature=DEMO_TEMPERATURE)


def _demo_predict(image: Image.Image) -> dict[str, Any]:
    features = feats.extract_features(image)
    quality = feats.assess_quality(features)

    scores = _heuristic_scores(features)
    class_names, probabilities = _demo_probabilities(scores)

    order = np.argsort(probabilities)[::-1]
    top_index = int(order[0])
    top_class = class_names[top_index]
    top_confidence = float(probabilities[top_index])

    # The heuristic must never sound more certain than it is.
    top_confidence = min(top_confidence, 0.88)

    notes = [
        "DEMO MODE: this result comes from a colour and texture heuristic, not a trained "
        "neural network. It is illustrative only.",
        "Measured evidence: "
        f"green {features.green_ratio * 100:.0f} %, yellow {features.yellow_ratio * 100:.0f} %, "
        f"brown/red {features.brown_ratio * 100:.0f} %, dark {features.dark_ratio * 100:.0f} %, "
        f"bleached {features.bleached_ratio * 100:.0f} %.",
    ]

    if not feats.looks_like_plant(features):
        notes.append(
            "Very little plant tissue was detected in this image. Make sure the photo shows a "
            "sugarcane leaf or stalk filling most of the frame."
        )
        top_class = "unknown"
        top_confidence = min(top_confidence, 0.25)
    elif quality["score"] < 0.45:
        notes.append("Image quality is low, so confidence has been reduced.")
        top_confidence *= 0.7
    elif top_confidence < CONFIDENCE_FLOOR:
        notes.append(
            f"No class reached the {CONFIDENCE_FLOOR * 100:.0f} % confidence floor, so the result "
            "is reported as unknown rather than guessed."
        )
        top_class = "unknown"

    return {
        "condition": top_class,
        "confidence": round(float(top_confidence), 4),
        "probabilities": {name: round(float(probabilities[i]), 4) for i, name in enumerate(class_names)},
        "features": features.as_dict(),
        "quality": quality,
        "model_source": "demo_heuristic",
        "model_label": "Demo colour and texture heuristic (no trained model loaded)",
        "is_demo": True,
        "notes": notes,
    }


# ------------------------------------------------------------- trained scoring
def _prepare_for_model(image: Image.Image) -> np.ndarray:
    size = _meta.get("input_size", [224, 224])
    # BILINEAR for the same reason as the soil model: training resized with an
    # aliased bilinear filter, so LANCZOS at serving time is a train/serve
    # mismatch that sharpens grain the model never saw. The measured gain here
    # is small (87.0 % vs 85.0 % on 200 training-pool images, so directional
    # rather than conclusive), but it is never worse and it matches training.
    resized = image.resize((int(size[1]), int(size[0])), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    if _meta.get("preprocessing") == "rescale_outside_model":
        array = array / 255.0
    return np.expand_dims(array, axis=0)


def _model_predict(image: Image.Image) -> dict[str, Any]:
    model = _load_model()
    features = feats.extract_features(image)
    quality = feats.assess_quality(features)

    batch = _prepare_for_model(image)
    raw = model.predict(batch, verbose=0)[0]
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 1:
        raw = raw.reshape(-1)
    # Convert logits to probabilities if the model does not end in softmax.
    if raw.min() < 0 or not np.isclose(raw.sum(), 1.0, atol=0.05):
        raw = _softmax(raw)

    class_names = list(_meta.get("class_names", DEFAULT_CLASS_NAMES))
    if len(class_names) != len(raw):
        class_names = [f"class_{i}" for i in range(len(raw))]

    top_index = int(np.argmax(raw))
    top_class = class_names[top_index]
    top_confidence = float(raw[top_index])

    notes = [
        "Prediction produced by the trained Keras model in models/.",
        f"Model reported {top_confidence * 100:.1f} % confidence for this class.",
    ]
    if quality["score"] < 0.45:
        notes.append("Image quality is low, which reduces how much this prediction should be trusted.")
    if top_confidence < CONFIDENCE_FLOOR:
        notes.append(
            f"Confidence is below the {CONFIDENCE_FLOOR * 100:.0f} % floor, so this is reported "
            "as unknown rather than presented as a diagnosis."
        )
        top_class = "unknown"

    return {
        "condition": top_class,
        "confidence": round(top_confidence, 4),
        "probabilities": {name: round(float(raw[i]), 4) for i, name in enumerate(class_names)},
        "features": features.as_dict(),
        "quality": quality,
        "model_source": "trained_model",
        "model_label": "Trained Keras image classifier",
        "is_demo": False,
        "notes": notes,
    }


def predict(image: Image.Image) -> dict[str, Any]:
    """Classify a sugarcane image, using the trained model when one is available."""
    if _load_model() is not None:
        try:
            return _model_predict(image)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Trained disease model failed, falling back to demo heuristic: %s", exc)
            result = _demo_predict(image)
            result["notes"].insert(0, f"The trained model failed on this image ({exc}). Demo heuristic used instead.")
            return result
    return _demo_predict(image)


# --------------------------------------------------------------- severity maths
def estimate_severity(condition: str, features: dict[str, Any], confidence: float) -> tuple[str, float, str]:
    """Return (severity, health_score, note).

    Severity is derived from how much of the visible tissue looks affected, not
    from the class alone, so a mild rust and a severe rust are distinguishable.
    """
    if condition == "unknown":
        return (
            "undetermined",
            50.0,
            "Severity cannot be estimated because the condition itself could not be determined.",
        )

    affected = float(
        features.get("brown_ratio", 0.0)
        + features.get("yellow_ratio", 0.0)
        + features.get("dark_ratio", 0.0)
        + features.get("bleached_ratio", 0.0)
    )
    green = float(features.get("green_ratio", 0.0))
    affected = min(1.0, max(0.0, affected))

    if condition == "healthy":
        health = min(100.0, 55.0 + green * 60.0 - affected * 40.0)
        return (
            "none",
            round(max(0.0, health), 1),
            "No disease symptoms were detected, so no severity rating applies.",
        )

    if affected < 0.15:
        severity = "mild"
        note = f"About {affected * 100:.0f} % of the visible tissue shows symptoms - early stage."
    elif affected < 0.35:
        severity = "moderate"
        note = f"About {affected * 100:.0f} % of the visible tissue shows symptoms - established infection."
    else:
        severity = "severe"
        note = f"About {affected * 100:.0f} % of the visible tissue shows symptoms - advanced infection."

    if confidence < 0.55:
        note += " Confidence is moderate, so confirm by inspecting several plants in the field."

    health = max(0.0, min(100.0, 92.0 - affected * 95.0))
    return severity, round(health, 1), note
