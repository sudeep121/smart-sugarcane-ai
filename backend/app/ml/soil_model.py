"""Soil image analysis.

Same honest two-mode design as the disease model. In demo mode this is a colour
and texture estimate of what the soil *looks like* - never a claim about NPK,
pH, EC or micronutrients, which cannot be measured from a photograph.
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

MODEL_STEMS = ("sugarcane_soil_model", "soil_model")
META_FILENAME = "soil_model_meta.json"

DEFAULT_CLASS_NAMES = ["black", "red", "sandy", "clay", "loamy"]
CONFIDENCE_FLOOR = 0.38

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
    for path in (meta_path, settings.project_root / "ml" / "soil_analysis" / "class_names.json"):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                names = data.get("class_names") if isinstance(data, dict) else data
                if isinstance(names, list) and names:
                    return [str(n) for n in names]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read %s: %s", path, exc)
    return list(DEFAULT_CLASS_NAMES)



def _accuracy_note() -> str:
    holdout = _meta.get("holdout_evaluation") or {}
    accuracy = holdout.get("accuracy")
    count = holdout.get("images")
    if accuracy is not None and count:
        return (
            f"Accuracy on {count} held-out photographs the model never saw during "
            f"training: {accuracy * 100:.1f} %"
        )
    value = _meta.get("val_accuracy")
    if value is None:
        return "Accuracy: not recorded for this model."
    return (
        f"Validation accuracy: {float(value) * 100:.1f} % - measured on the split that "
        "also selected the best epoch, so treat it as optimistic."
    )

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
            _load_error = f"No trained soil model found in {settings.models_dir}."
            return None

        try:
            import tensorflow as tf  # noqa: PLC0415

            _model = model_runtime.load(path)
        except ImportError:
            _load_error = (
                "A soil model file exists but TensorFlow is not installed. "
                "Install it with: pip install -r ml/disease_detection/requirements.txt"
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
        logger.info("Loaded trained soil model from %s", path)
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
                _load_error or "No trained soil model present.",
                "Soil analysis is running as a DEMO visual estimate. Accuracy depends on "
                "lighting and image quality.",
                "A photograph cannot measure NPK, pH, electrical conductivity or micronutrients.",
                "To train a real model see docs/DATASET_GUIDE.md and run ml/soil_analysis/train.py",
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
            # Prefer the held-out figure. val_accuracy is measured on the split
            # that also drove EarlyStopping and checkpoint selection, so it
            # flatters the model; holdout_evaluation never influenced training.
            _accuracy_note(),
            "Even a trained image model classifies APPEARANCE only. Laboratory soil testing "
            "provides accurate nutrient and pH values.",
        ],
    }


# ----------------------------------------------------------------- demo scoring
def _membership(value: float, low: float, high: float, softness: float = 0.12) -> float:
    """1.0 inside [low, high], falling off smoothly outside it."""
    if low <= value <= high:
        return 1.0
    distance = low - value if value < low else value - high
    return float(max(0.0, 1.0 - distance / max(softness, 1e-6)))


def _hue_membership(hue: float, low: float, high: float, softness: float = 14.0) -> float:
    return _membership(hue, low, high, softness)


DEMO_TEMPERATURE = 0.18


def _softmax(scores: np.ndarray, temperature: float = DEMO_TEMPERATURE) -> np.ndarray:
    shifted = (scores - scores.max()) / max(temperature, 1e-6)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


#: Per class: (attribute, low, high, softness, weight).
#: Scores are a WEIGHTED MEAN of the memberships rather than a sum of bonuses.
#: With a sum, a class could win on two easy criteria while completely failing a
#: decisive one - a near-black soil scored as clay because "smooth" and
#: "low saturation" both paid out while the brightness term merely contributed
#: nothing. A weighted mean lets a failed criterion drag the whole score down.
_CRITERIA: dict[str, list[tuple[str, float, float, float, float]]] = {
    "black": [
        ("value", 0.00, 0.33, 0.10, 3.0),
        ("saturation", 0.00, 0.45, 0.15, 1.0),
    ],
    # Hue is what actually separates red soil from a dark brown loam - both are
    # "brownish", but iron-rich red soil sits below ~24 deg while loam sits above
    # it. Saturation reinforces this: lateritic red is vivid, loam is muted.
    "red": [
        ("hue", 2.0, 24.0, 14.0, 3.0),
        ("saturation", 0.42, 1.00, 0.12, 2.0),
        ("value", 0.22, 0.72, 0.12, 1.5),
    ],
    "sandy": [
        ("hue", 25.0, 58.0, 14.0, 2.0),
        ("value", 0.48, 1.00, 0.10, 3.0),
        ("saturation", 0.00, 0.55, 0.12, 1.0),
        ("grain", 0.50, 1.50, 0.30, 1.5),
    ],
    "clay": [
        ("hue", 10.0, 42.0, 14.0, 1.5),
        ("value", 0.30, 0.62, 0.10, 3.0),
        ("saturation", 0.00, 0.42, 0.12, 1.5),
        ("grain", 0.00, 0.35, 0.25, 1.5),
    ],
    # Loam is crumbly, so anything from moderate to strongly granular fits. An
    # earlier upper bound of 0.90 excluded ordinary crumbly loam and let red soil
    # win on hue alone.
    "loamy": [
        ("hue", 18.0, 40.0, 14.0, 1.5),
        ("value", 0.28, 0.58, 0.10, 3.0),
        ("saturation", 0.18, 0.55, 0.12, 2.0),
        ("grain", 0.25, 1.50, 0.30, 1.0),
    ],
}


def _heuristic_scores(f: feats.ImageFeatures) -> dict[str, float]:
    attributes = {
        "hue": f.mean_hue,
        "saturation": f.mean_saturation,
        "value": f.mean_value,
        "grain": min(f.graininess * 12.0, 1.5),
    }

    scores: dict[str, float] = {}
    for soil_class, criteria in _CRITERIA.items():
        total_weight = sum(weight for *_, weight in criteria)
        earned = 0.0
        for attribute, low, high, softness, weight in criteria:
            value = attributes[attribute]
            fit = (
                _hue_membership(value, low, high, softness)
                if attribute == "hue"
                else _membership(value, low, high, softness)
            )
            earned += weight * fit
        scores[soil_class] = earned / total_weight
    return scores


def _texture_appearance(f: feats.ImageFeatures) -> str:
    grain = f.graininess * 12.0
    if grain >= 0.75:
        return "Coarse and grainy - individual particles are visible"
    if grain >= 0.35:
        return "Medium, crumbly appearance"
    return "Fine and smooth - few individual particles visible"


def _moisture_appearance(f: feats.ImageFeatures, soil_type: str) -> str:
    """Dark soils are naturally dark, so the thresholds shift by class."""
    value = f.mean_value
    shift = -0.10 if soil_type in {"black", "clay"} else 0.0
    if value < 0.26 + shift:
        return "wet"
    if value < 0.44 + shift:
        return "moderate"
    return "dry"


def _organic_matter_appearance(f: feats.ImageFeatures, soil_type: str) -> str:
    value = f.mean_value
    if soil_type == "black":
        # Black soil is dark from its mineralogy, not necessarily from organic carbon.
        return "moderate"
    if value < 0.30:
        return "high"
    if value < 0.48:
        return "moderate"
    return "low"


#: Below this mean saturation the image has no meaningful colour cast, so naming
#: one would be misleading - a near-grey soil should not be called "yellowish"
#: just because a handful of faintly tinted grains dominate the hue average.
MIN_SATURATION_FOR_CAST = 0.16


def _colour_description(rgb: list[int], hue: float, value: float, saturation: float) -> str:
    r, g, b = rgb
    if value < 0.25:
        base = "very dark grey to near black"
    elif value < 0.4:
        base = "dark brown"
    elif value < 0.6:
        base = "medium brown"
    else:
        base = "pale light brown"

    is_neutral = saturation < MIN_SATURATION_FOR_CAST or (abs(r - g) < 12 and abs(g - b) < 12)
    if is_neutral:
        base += " with little colour, close to neutral grey"
    elif 2 <= hue <= 25 and r > g + 20:
        base += " with a distinctly reddish cast"
    elif 26 <= hue <= 55:
        base += " with a yellowish cast"

    return f"Approximately {base} (dominant RGB {r}, {g}, {b})"


def _demo_predict(image: Image.Image) -> dict[str, Any]:
    features = feats.extract_features(image)
    quality = feats.assess_quality(features)

    scores = _heuristic_scores(features)
    class_names = list(scores)
    probabilities = _softmax(np.array([scores[n] for n in class_names], dtype=np.float64))

    top_index = int(np.argmax(probabilities))
    top_class = class_names[top_index]
    top_confidence = min(float(probabilities[top_index]), 0.82)

    notes = [
        "DEMO visual estimate - accuracy depends on lighting and image quality.",
        "Measured evidence: "
        f"mean hue {features.mean_hue:.0f} deg, saturation {features.mean_saturation:.2f}, "
        f"brightness {features.mean_value:.2f}, graininess {features.graininess:.3f}.",
    ]

    if not feats.looks_like_soil(features):
        notes.append(
            f"This image is {features.green_ratio * 100:.0f} % green, which suggests vegetation "
            "rather than bare soil. Photograph a freshly turned patch of bare soil instead."
        )
        top_class = "mixed"
        top_confidence = min(top_confidence, 0.25)
    elif quality["score"] < 0.45:
        notes.append("Image quality is low, so confidence has been reduced.")
        top_confidence *= 0.7
        if top_confidence < CONFIDENCE_FLOOR:
            top_class = "mixed"
    elif top_confidence < CONFIDENCE_FLOOR:
        notes.append(
            f"No soil class reached the {CONFIDENCE_FLOOR * 100:.0f} % confidence floor, so the "
            "result is reported as Mixed / Unknown rather than guessed."
        )
        top_class = "mixed"

    probability_map = {name: round(float(probabilities[i]), 4) for i, name in enumerate(class_names)}
    probability_map.setdefault("mixed", 0.0)

    return {
        "soil_type": top_class,
        "confidence": round(float(top_confidence), 4),
        "probabilities": probability_map,
        "features": features.as_dict(),
        "quality": quality,
        "texture_appearance": _texture_appearance(features),
        "moisture_appearance": _moisture_appearance(features, top_class),
        "organic_matter_appearance": _organic_matter_appearance(features, top_class),
        "colour_description": _colour_description(
            features.dominant_rgb, features.mean_hue, features.mean_value, features.mean_saturation
        ),
        "dominant_rgb": features.dominant_rgb,
        "model_source": "demo_heuristic",
        "model_label": "Demo visual estimate (colour and texture heuristic)",
        "is_demo": True,
        "notes": notes,
    }


# ------------------------------------------------------------- trained scoring
def _prepare_for_model(image: Image.Image) -> np.ndarray:
    size = _meta.get("input_size", [224, 224])
    # BILINEAR, not LANCZOS. Pillow antialiases when downscaling, and on a
    # 1280px field photo reduced to 192px that matters: LANCZOS sharpens the
    # grain and cost 3 points of held-out accuracy, taking red soil recall from
    # 100 % down to 91 % and misreading a real red field photo as alluvial.
    # Measured on 100 held-out images: BILINEAR 93.0 %, BICUBIC 91.0 %,
    # BOX/HAMMING/LANCZOS 90.0 %.
    resized = image.resize((int(size[1]), int(size[0])), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    if _meta.get("preprocessing") == "rescale_outside_model":
        array = array / 255.0
    return np.expand_dims(array, axis=0)


def _model_predict(image: Image.Image) -> dict[str, Any]:
    model = _load_model()
    features = feats.extract_features(image)
    quality = feats.assess_quality(features)

    raw = np.asarray(model.predict(_prepare_for_model(image), verbose=0)[0], dtype=np.float64).reshape(-1)
    if raw.min() < 0 or not np.isclose(raw.sum(), 1.0, atol=0.05):
        raw = _softmax(raw, temperature=1.0)

    class_names = list(_meta.get("class_names", DEFAULT_CLASS_NAMES))
    if len(class_names) != len(raw):
        class_names = [f"class_{i}" for i in range(len(raw))]

    top_index = int(np.argmax(raw))
    top_class = class_names[top_index]
    top_confidence = float(raw[top_index])

    notes = [
        "Prediction produced by the trained Keras soil model in models/.",
        "Even a trained model classifies soil APPEARANCE. It cannot measure NPK, pH or EC.",
    ]
    if top_confidence < CONFIDENCE_FLOOR:
        notes.append("Confidence is below the floor, so the result is reported as Mixed / Unknown.")
        top_class = "mixed"

    probability_map = {name: round(float(raw[i]), 4) for i, name in enumerate(class_names)}
    probability_map.setdefault("mixed", 0.0)

    return {
        "soil_type": top_class,
        "confidence": round(top_confidence, 4),
        "probabilities": probability_map,
        "features": features.as_dict(),
        "quality": quality,
        "texture_appearance": _texture_appearance(features),
        "moisture_appearance": _moisture_appearance(features, top_class),
        "organic_matter_appearance": _organic_matter_appearance(features, top_class),
        "colour_description": _colour_description(
            features.dominant_rgb, features.mean_hue, features.mean_value, features.mean_saturation
        ),
        "dominant_rgb": features.dominant_rgb,
        "model_source": "trained_model",
        "model_label": "Trained Keras soil classifier",
        "is_demo": False,
        "notes": notes,
    }


def predict(image: Image.Image) -> dict[str, Any]:
    if _load_model() is not None:
        try:
            return _model_predict(image)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Trained soil model failed, falling back to demo heuristic: %s", exc)
            result = _demo_predict(image)
            result["notes"].insert(0, f"The trained model failed on this image ({exc}). Demo estimate used instead.")
            return result
    return _demo_predict(image)
