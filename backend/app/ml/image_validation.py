"""Gate uploads before any analysis runs.

The models in this project are closed-set classifiers: the disease model always
returns one of five sugarcane conditions, the soil model always returns one of
four soil types. Neither has a "none of these" output, so given a photograph of
a person they will still answer, and answer confidently - a portrait was
returned as "Clay Soil, 62 % confidence". A closed-set classifier cannot refuse;
something in front of it has to.

Validation runs in two stages, because "is this a plant at all" and "is this
plant sugarcane" are genuinely different problems:

Stage A - measurable, no model required. Colour composition separates
vegetation, bare soil and everything else (people, buildings, screenshots,
food, objects) reliably enough to reject the obvious cases. Calibrated on 250
sugarcane, 250 maize and 250 soil photographs.

Stage B - the trained validator model. Stage A cannot tell sugarcane from
maize: measured on those samples, vegetation cover is 0.57 median for sugarcane
and 0.66 for maize, which overlaps almost completely. Long green blade-shaped
leaves are long green blade-shaped leaves. Only a model trained with maize as
an explicit negative can make that call, so if the validator model is missing,
say so rather than pretending Stage A settled it.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import settings
from app.ml import model_runtime

logger = logging.getLogger(__name__)

MODEL_STEMS = ("sugarcane_validator_model", "validator_model")
META_FILENAME = "validator_model_meta.json"
POSITIVE_CLASS = "sugarcane"

# --- Stage A thresholds, from the calibration above ---------------------------
# Sugarcane photographs sit at 0.26 vegetation cover on the 10th percentile, so
# 0.15 leaves headroom for a close crop of a diseased, largely brown leaf.
MIN_VEGETATION_FOR_PLANT = 0.15
# Two thresholds, because "not soil" and "cannot tell" deserve different
# answers. Measured on 300 soil photographs: at or above 0.50 soil-coloured
# accepts 93 %, the 0.30-0.50 band holds 3 % (analysed as genuinely ambiguous
# rather than guessed at), and below 0.30 rejects 4 %. The portrait that was
# returned as "Clay Soil, 62 %" scores 0.18.
SOIL_CONFIDENT_FRACTION = 0.50
MIN_SOIL_FRACTION = 0.30
# Above this, a "soil" upload is really a plant photograph.
MAX_VEGETATION_FOR_SOIL = 0.25
# Mostly unsaturated pixels means a screenshot, document or studio backdrop.
FLAT_IMAGE_FRACTION = 0.55
# Below this the validator model is not confident enough to be trusted either way.
VALIDATOR_CONFIDENCE_FLOOR = 0.60
MIN_USABLE_PIXELS = 64

_lock = threading.Lock()
_model = None
_meta: dict[str, Any] = {}
_load_attempted = False
_load_error: str | None = None


@dataclass
class ContentProfile:
    """Coarse colour composition of an image."""

    vegetation: float
    soil_like: float
    flat: float
    brightness: float
    contrast: float
    width: int
    height: int
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "vegetation_fraction": round(self.vegetation, 4),
            "soil_fraction": round(self.soil_like, 4),
            "flat_fraction": round(self.flat, 4),
            "brightness": round(self.brightness, 4),
            "contrast": round(self.contrast, 4),
            "width": self.width,
            "height": self.height,
            **self.extras,
        }


def profile_image(image: Image.Image) -> ContentProfile:
    """Colour composition of the image, in fractions of total pixels."""
    width, height = image.size
    small = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    array = np.asarray(small, dtype=np.float32) / 255.0
    red, green, blue = array[..., 0], array[..., 1], array[..., 2]
    high = array.max(axis=2)
    low = array.min(axis=2)
    delta = high - low
    saturation = np.where(high > 1e-6, delta / np.maximum(high, 1e-6), 0.0)

    hue = np.zeros_like(high)
    nonzero = delta > 1e-6
    mask = (high == red) & nonzero
    hue[mask] = ((green - blue)[mask] / delta[mask]) % 6
    mask = (high == green) & nonzero
    hue[mask] = ((blue - red)[mask] / delta[mask]) + 2
    mask = (high == blue) & nonzero
    hue[mask] = ((red - green)[mask] / delta[mask]) + 4
    hue *= 60.0

    vegetation = (hue >= 55) & (hue <= 190) & (saturation >= 0.15) & (high >= 0.10)
    # Soil reflects red at least as strongly as blue. Near-black pixels are kept
    # as soil-plausible because black soil is dark and unsaturated, which would
    # otherwise read as "no colour information" and reject a valid photograph.
    warm = red >= blue - 0.015
    earthy = warm & (saturation >= 0.10) & ((hue <= 64) | (hue >= 348))
    dark = warm & (high <= 0.30)

    return ContentProfile(
        vegetation=float(vegetation.mean()),
        soil_like=float((earthy | dark).mean()),
        flat=float((saturation < 0.10).mean()),
        brightness=float(high.mean()),
        contrast=float(high.std()),
        width=width,
        height=height,
    )


# --------------------------------------------------------------- validator model
def find_model_file() -> Path | None:
    """Preferring .onnx, which serves in a fifth of the memory Keras needs."""
    return model_runtime.find_model_file(settings.models_dir, MODEL_STEMS)


def _load_model():
    global _model, _meta, _load_attempted, _load_error
    if _load_attempted:
        return _model
    with _lock:
        if _load_attempted:
            return _model
        _load_attempted = True
        path = find_model_file()
        if path is None:
            _load_error = "No validator model file in models/."
            return None
        try:
            _model = model_runtime.load(path)
            meta_path = settings.models_dir / META_FILENAME
            if meta_path.exists():
                _meta = json.loads(meta_path.read_text(encoding="utf-8"))
            logger.info("Loaded sugarcane validator model from %s", path)
        except Exception as exc:  # noqa: BLE001
            _model = None
            _load_error = str(exc)
            logger.warning("Validator model failed to load: %s", exc)
        return _model


def reload_model() -> None:
    global _model, _meta, _load_attempted, _load_error
    with _lock:
        _model = None
        _meta = {}
        _load_attempted = False
        _load_error = None


def model_available() -> bool:
    return _load_model() is not None


def _validator_scores(image: Image.Image) -> dict[str, float] | None:
    model = _load_model()
    if model is None:
        return None
    size = _meta.get("input_size", [160, 160])
    resized = image.convert("RGB").resize(
        (int(size[1]), int(size[0])), Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32)
    if _meta.get("preprocessing") == "rescale_outside_model":
        array = array / 255.0
    raw = np.asarray(model.predict(array[None], verbose=0)[0], dtype=np.float64).reshape(-1)
    if raw.min() < 0 or not np.isclose(raw.sum(), 1.0, atol=0.05):
        shifted = np.exp(raw - raw.max())
        raw = shifted / shifted.sum()
    names = list(_meta.get("class_names") or [])
    if len(names) != len(raw):
        names = [f"class_{i}" for i in range(len(raw))]
    return {name: float(value) for name, value in zip(names, raw)}


# ------------------------------------------------------------------- public API
def _result(
    is_valid: bool,
    confidence: float,
    image_type: str,
    message: str,
    *,
    stage: str,
    profile: ContentProfile,
    checked_by: str,
    title: str | None = None,
    scores: dict[str, float] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "isSugarcane": bool(is_valid),
        "confidence": round(float(confidence), 4),
        "imageType": image_type,
        "title": title or ("Image accepted" if is_valid else "Invalid image"),
        "message": message,
        "stage": stage,
        "checked_by": checked_by,
        "profile": profile.as_dict(),
        "notes": notes or [],
    }
    if scores is not None:
        payload["scores"] = {name: round(value, 4) for name, value in scores.items()}
    return payload


def _describe_non_plant(profile: ContentProfile) -> tuple[str, str]:
    """Name what a clearly-non-plant image looks like, for a useful message."""
    if profile.soil_like >= 0.55:
        return (
            "soil_only",
            "This looks like a photograph of soil, not a sugarcane plant. "
            "Soil-only images cannot be analysed in Plant Health - use the Soil "
            "Analysis section for soil photographs.",
        )
    if profile.flat >= FLAT_IMAGE_FRACTION:
        return (
            "not_a_plant",
            "This looks like a screenshot, document or an object on a plain "
            "background rather than a sugarcane plant.",
        )
    return (
        "not_a_plant",
        "No plant foliage was detected in this image.",
    )


def validate_sugarcane_image(image: Image.Image) -> dict[str, Any]:
    """Decide whether an upload may proceed to sugarcane plant analysis."""
    profile = profile_image(image)
    notes: list[str] = []

    if min(profile.width, profile.height) < MIN_USABLE_PIXELS:
        return _result(
            False, 0.99, "unclear",
            f"This image is only {profile.width} by {profile.height} pixels, which is "
            "too small to identify a plant. Please upload a larger photo.",
            title="Unable to Identify Sugarcane",
            stage="A", profile=profile, checked_by="rule_engine",
        )

    # Stage A - is this vegetation at all?
    if profile.vegetation < MIN_VEGETATION_FOR_PLANT:
        image_type, message = _describe_non_plant(profile)
        shortfall = 1.0 - (profile.vegetation / MIN_VEGETATION_FOR_PLANT)
        return _result(
            False, 0.55 + 0.44 * shortfall, image_type, message,
            title="Sugarcane Photo Required",
            stage="A", profile=profile, checked_by="rule_engine",
            notes=[
                f"Only {profile.vegetation * 100:.0f} % of the image is plant foliage; "
                f"sugarcane photographs are typically 26 % or more.",
            ],
        )

    # Stage B - it is a plant, but is it sugarcane?
    scores = _validator_scores(image)
    if scores is None:
        notes.append(
            "The trained sugarcane validator is not installed, so this image was "
            "only checked for plant foliage. Maize, rice, grass and other "
            "blade-leaved crops cannot be distinguished from sugarcane without it."
        )
        return _result(
            True, 0.5, "plant_unverified",
            "Foliage detected. This image was NOT confirmed to be sugarcane - the "
            "validator model is not installed, so treat any result with caution.",
            title="Not confirmed as sugarcane",
            stage="A", profile=profile, checked_by="rule_engine", notes=notes,
        )

    positive = scores.get(POSITIVE_CLASS, 0.0)
    top_name = max(scores, key=scores.get)
    top_score = scores[top_name]

    if top_name == POSITIVE_CLASS and positive >= VALIDATOR_CONFIDENCE_FLOOR:
        return _result(
            True, positive, "sugarcane_plant",
            "Sugarcane detected. Image can be analysed.",
            title="Sugarcane detected",
            stage="B", profile=profile, checked_by="trained_model", scores=scores,
        )

    if top_score < VALIDATOR_CONFIDENCE_FLOOR:
        return _result(
            False, top_score, "unclear",
            "Sugarcane could not be reliably identified in this image. Please upload "
            "a clearer, well-lit photo showing the leaves and stem.",
            title="Unable to Identify Sugarcane",
            stage="B", profile=profile, checked_by="trained_model", scores=scores,
            notes=["Neither class reached the confidence floor, so no guess is made."],
        )

    return _result(
        False, top_score, "other_plant",
        "This looks like a plant, but not sugarcane. Please upload a photo of a "
        "sugarcane plant, showing the leaves and stem.",
        title="Sugarcane Photo Required",
        stage="B", profile=profile, checked_by="trained_model", scores=scores,
    )


def validate_soil_image(image: Image.Image) -> dict[str, Any]:
    """Decide whether an upload may proceed to soil analysis.

    Three outcomes, not two. "This is not soil" and "I cannot tell whether
    there is enough soil here" are different situations and get different
    messages, because the second one is fixed by taking a better photograph
    while the first is fixed by photographing something else entirely.

    Nothing about being uploaded through the Soil Analysis section makes an
    image soil - the pixels are what decide.
    """
    profile = profile_image(image)

    invalid_detail = (
        "For accurate soil analysis, upload a clear photo showing the soil "
        "from your agricultural field."
    )

    if min(profile.width, profile.height) < MIN_USABLE_PIXELS:
        return _result(
            False, 0.99, "unclear",
            invalid_detail,
            title="Soil Not Clearly Visible - Please upload a clear photo of the soil.",
            stage="A", profile=profile, checked_by="rule_engine",
            notes=[
                f"The image is only {profile.width} by {profile.height} pixels, "
                "which is too small to judge soil colour or texture.",
            ],
        )

    # A plant photograph. Checked before soil colour because foliage photos
    # often include soil in the background: sugarcane images measure 0.40
    # soil-coloured on average, which on colour alone would sail through.
    if profile.vegetation > MAX_VEGETATION_FOR_SOIL and profile.vegetation > profile.soil_like:
        return _result(
            False, min(0.99, 0.5 + profile.vegetation), "plant_photo",
            invalid_detail,
            title="Invalid Image - Please upload only a soil photo.",
            stage="A", profile=profile, checked_by="rule_engine",
            notes=[
                f"{profile.vegetation * 100:.0f} % of this image is plant foliage. "
                "For soil analysis, photograph the bare soil surface with the crop "
                "out of the frame.",
            ],
        )

    # Colour alone is not enough here. A rust-infected sugarcane leaf is brown,
    # so it scored 0.53 soil-coloured against 0.43 foliage and passed the check
    # above - while the trained validator called it sugarcane at 97.9 %. Ask the
    # model that was trained to answer this question.
    scores = _validator_scores(image)
    if scores is not None:
        cane_score = scores.get(POSITIVE_CLASS, 0.0)
        if cane_score >= VALIDATOR_CONFIDENCE_FLOOR:
            return _result(
                False, cane_score, "plant_photo",
                invalid_detail,
                title="Invalid Image - Please upload only a soil photo.",
                stage="B", profile=profile, checked_by="trained_model", scores=scores,
                notes=[
                    "The trained validator identified a sugarcane plant in this "
                    f"photo ({cane_score * 100:.0f} % confidence). For soil analysis, "
                    "photograph the bare soil surface with the crop out of the frame.",
                ],
            )

    if profile.soil_like < MIN_SOIL_FRACTION:
        detail = (
            "screenshot, document or an object on a plain background"
            if profile.flat >= FLAT_IMAGE_FRACTION
            else "subject other than soil"
        )
        shortfall = 1.0 - (profile.soil_like / MIN_SOIL_FRACTION)
        return _result(
            False, 0.55 + 0.44 * shortfall, "not_soil",
            invalid_detail,
            title="Invalid Image - Please upload only a soil photo.",
            stage="A", profile=profile, checked_by="rule_engine",
            notes=[
                f"Only {profile.soil_like * 100:.0f} % of this image has soil-like "
                f"colour, and it looks like a {detail}. Soil photographs are "
                "typically 74 % or more.",
            ],
        )

    # Some soil, but not enough of the frame to be confident. Do not guess.
    if profile.soil_like < SOIL_CONFIDENT_FRACTION:
        return _result(
            False, 0.5 + 0.3 * (SOIL_CONFIDENT_FRACTION - profile.soil_like), "unclear_soil",
            "Move closer so the soil surface fills the frame, and keep plants, "
            "hands, tools and shadows out of the picture.",
            title="Soil Not Clearly Visible - Please upload a clear photo of the soil.",
            stage="A", profile=profile, checked_by="rule_engine",
            notes=[
                f"Only {profile.soil_like * 100:.0f} % of this image is soil, which is "
                "not enough to analyse reliably. No estimate is made rather than "
                "guessing from a partial view.",
            ],
        )

    return _result(
        True, min(0.99, profile.soil_like), "soil",
        "Soil image verified. Starting analysis...",
        title="Soil image verified",
        stage="A", profile=profile, checked_by="rule_engine",
    )


def status() -> dict[str, Any]:
    model = _load_model()
    meta = dict(_meta)
    return {
        "validator_model_available": model is not None,
        "model_path": str(find_model_file()) if find_model_file() else None,
        "load_error": _load_error,
        "class_names": meta.get("class_names"),
        "holdout": meta.get("holdout_evaluation"),
        "notes": [
            "Stage A (colour composition) rejects non-plant and non-soil images "
            "without a model.",
            "Stage B (trained validator) is what separates sugarcane from maize "
            "and other blade-leaved crops. Without it, that distinction is not made.",
        ],
    }
