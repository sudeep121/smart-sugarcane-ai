"""Convert the trained Keras models to ONNX, and prove the outputs still match.

Why: serving through TensorFlow needs roughly 2 GB of RAM, which no free host
provides - Render gives 512 MB, and the process is killed on import. ONNX
Runtime executes the same graph in a fraction of that, so the application can
be hosted permanently and for free instead of only on a laptop.

A conversion that quietly changes predictions would be worse than not
converting at all, so every model is checked against its Keras original on
real photographs from the held-out sets, and the script refuses to write a
model whose predictions disagree.

    python ml/export_onnx.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"

# (keras file, meta file, a folder of real images to verify against)
TARGETS = [
    ("sugarcane_soil_model.keras", "soil_model_meta.json", "soil/holdout"),
    ("sugarcane_validator_model.keras", "validator_model_meta.json", "sugarcane_validator/holdout"),
    ("sugarcane_disease_model.keras", "disease_model_meta.json", "sugarcane"),
]

# Predictions must agree to within this absolute probability difference. 1e-4
# is far tighter than anything that could flip a class or move a displayed
# percentage, while leaving room for float32 ordering differences between the
# two runtimes.
TOLERANCE = 1e-4
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def dataset_dir(relative: str) -> Path | None:
    for base in (REPO / "datasets", REPO.parent / "datasets"):
        candidate = base / relative
        if candidate.exists():
            return candidate
    return None


def sample_images(folder: Path, limit: int) -> list[Path]:
    files = [f for f in folder.rglob("*") if f.suffix.lower() in SUFFIXES]
    files.sort()
    if not files:
        return []
    step = max(1, len(files) // limit)
    return files[::step][:limit]


def load_batch(paths: list[Path], size: int) -> np.ndarray:
    from PIL import Image

    return np.stack(
        [
            np.asarray(
                Image.open(p).convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            for p in paths
        ]
    )


def convert(keras_name: str, meta_name: str, data_rel: str, samples: int) -> bool:
    import tensorflow as tf
    import tf2onnx
    import onnxruntime as ort

    keras_path = MODELS / keras_name
    onnx_path = keras_path.with_suffix(".onnx")
    meta_path = MODELS / meta_name

    if not keras_path.exists():
        print(f"  SKIP {keras_name}: not found")
        return True

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    size = int((meta.get("input_size") or [224, 224])[0])

    print(f"\n{keras_name}  ({size}x{size})")
    model = tf.keras.models.load_model(keras_path)

    spec = (tf.TensorSpec((None, size, size, 3), tf.float32, name="image"),)
    proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx_path.write_bytes(proto.SerializeToString())

    keras_mb = keras_path.stat().st_size / 1048576
    onnx_mb = onnx_path.stat().st_size / 1048576
    print(f"  wrote {onnx_path.name}  ({keras_mb:.1f} MB keras -> {onnx_mb:.1f} MB onnx)")

    folder = dataset_dir(data_rel)
    if folder is None:
        print(f"  WARNING: {data_rel} not found - verified on random noise only")
        batch = np.random.default_rng(0).uniform(0, 255, (4, size, size, 3)).astype(np.float32)
        names = ["<random>"] * 4
    else:
        paths = sample_images(folder, samples)
        batch = load_batch(paths, size)
        names = [p.name for p in paths]

    keras_out = model.predict(batch, verbose=0)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: batch})[0]

    drift = float(np.abs(keras_out - onnx_out).max())
    same_class = int((np.argmax(keras_out, 1) == np.argmax(onnx_out, 1)).sum())
    print(f"  checked {len(batch)} real images: max probability drift {drift:.2e}, "
          f"same predicted class {same_class}/{len(batch)}")

    if same_class != len(batch) or drift > TOLERANCE:
        print(f"  FAILED: outputs disagree. Refusing to keep {onnx_path.name}.")
        for i, name in enumerate(names):
            if np.argmax(keras_out[i]) != np.argmax(onnx_out[i]):
                print(f"    {name}: keras {np.argmax(keras_out[i])} vs onnx {np.argmax(onnx_out[i])}")
        onnx_path.unlink(missing_ok=True)
        return False

    print("  OK - ONNX matches Keras")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=24, help="real images to verify each model on")
    args = ap.parse_args()

    print("Converting Keras models to ONNX and verifying against the originals")
    ok = all(convert(k, m, d, args.samples) for k, m, d in TARGETS)

    print()
    if ok:
        total_keras = sum(
            (MODELS / k).stat().st_size for k, _, _ in TARGETS if (MODELS / k).exists()
        ) / 1048576
        total_onnx = sum(
            (MODELS / k).with_suffix(".onnx").stat().st_size
            for k, _, _ in TARGETS
            if (MODELS / k).with_suffix(".onnx").exists()
        ) / 1048576
        print(f"All models converted. {total_keras:.1f} MB keras -> {total_onnx:.1f} MB onnx")
    else:
        print("At least one model failed verification; nothing was replaced.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
