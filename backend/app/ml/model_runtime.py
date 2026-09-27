"""Load an image model through whichever runtime is available.

The three image models were trained with Keras and serving them that way needs
TensorFlow, which measures 579 MB of resident memory once the models are loaded
and inference has run. Every free hosting tier caps at 512 MB, so the process
is killed on import and the application can only ever run on a laptop.

The same graphs exported to ONNX and served by ONNX Runtime peak at 117 MB -
about a fifth - which fits a free tier with room to spare. ml/export_onnx.py
performs the conversion and refuses to write a model whose predictions differ
from the Keras original; measured drift was under 5e-06 with identical
predicted classes on every held-out image tested.

So ONNX is preferred when present, and Keras is used when it is not. Local
development keeps working untouched, and a deployment ships without
TensorFlow at all. The wrapper deliberately mirrors the Keras call signature -
`predict(batch, verbose=0)` returning an array - so the calling code does not
need to know which runtime answered.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class OnnxModel:
    """Minimal stand-in for a Keras model, backed by ONNX Runtime."""

    runtime = "onnxruntime"

    def __init__(self, path: Path):
        import onnxruntime as ort  # noqa: PLC0415

        options = ort.SessionOptions()
        # Free tiers hand out a fraction of a core; spawning a thread per core
        # costs memory and contends with the web server for no gain.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self.path = path

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:  # noqa: ARG002
        """`verbose` exists only so this is a drop-in for model.predict()."""
        array = np.asarray(batch, dtype=np.float32)
        return self._session.run(None, {self._input: array})[0]

    @property
    def output_shape(self) -> tuple[Any, ...]:
        return tuple(self._session.get_outputs()[0].shape)


class KerasModel:
    """The original runtime, used when no ONNX export is present."""

    runtime = "tensorflow"

    def __init__(self, path: Path):
        from tensorflow import keras  # noqa: PLC0415

        self._model = keras.models.load_model(path)
        self.path = path

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        return self._model.predict(batch, verbose=verbose)

    @property
    def output_shape(self) -> tuple[Any, ...]:
        return tuple(self._model.output_shape)


def find_model_file(models_dir: Path, basenames: tuple[str, ...]) -> Path | None:
    """First existing model file, preferring .onnx over .keras.

    `basenames` are stems without an extension, in priority order.
    """
    for stem in basenames:
        for suffix in (".onnx", ".keras", ".h5"):
            candidate = models_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def load(path: Path) -> OnnxModel | KerasModel:
    """Load `path` with the runtime its extension implies.

    Raises whatever the underlying runtime raises; callers already treat a
    failure here as "no trained model" and fall back to their heuristics.
    """
    if path.suffix == ".onnx":
        model = OnnxModel(path)
    else:
        model = KerasModel(path)
    logger.info("Loaded %s via %s", path.name, model.runtime)
    return model
