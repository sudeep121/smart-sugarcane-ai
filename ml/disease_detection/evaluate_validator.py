"""Evaluate the sugarcane image validator on images it never saw during training.

Reports maize separately. Maize is the case that matters: if the model has
learned "long green blade leaf = sugarcane" rather than something specific to
sugarcane, maize is where that shows up, and a single overall accuracy figure
would hide it behind the easy negatives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def _resolve(relative: str) -> Path:
    """Locate a dataset folder without hard-coding one machine's layout."""
    for base in (REPO / "datasets", REPO.parent / "datasets"):
        candidate = base / relative
        if candidate.exists():
            return candidate
    return REPO / "datasets" / relative



def load(path: Path, size: int) -> np.ndarray:
    """Decode and resize exactly as the backend does.

    Pillow rather than tf.io.decode_image for two reasons: it matches the
    serving path, so this measures the model people actually use, and it reads
    the WEBP files that are scattered through these web-scraped datasets under
    a .jpg name - TensorFlow rejects those outright with "Unknown image file
    format" and the evaluation cannot finish.
    """
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def predict_dir(model, files, size, batch=48):
    out = []
    for i in range(0, len(files), batch):
        chunk = np.stack([load(f, size) for f in files[i : i + batch]])
        out.append(model.predict(chunk, verbose=0))
    return np.concatenate(out) if out else np.zeros((0, 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=REPO / "models" / "sugarcane_validator_model.onnx")
    ap.add_argument("--meta", type=Path, default=REPO / "models" / "validator_model_meta.json")
    ap.add_argument("--holdout", type=Path, default=_resolve("sugarcane_validator/holdout"))
    ap.add_argument("--limit", type=int, default=0, help="cap images per class (0 = all)")
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    names = meta["class_names"]
    size = int(meta["input_size"][0])
    pos_index = names.index("sugarcane")
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    class _Runner:
        @staticmethod
        def predict(batch, verbose=0):  # noqa: ARG004
            return session.run(None, {input_name: batch})[0]

    model = _Runner()

    print(f"HELD-OUT SET - never seen during training\n")
    totals = {}
    for label in names:
        files = sorted(f for f in (args.holdout / label).iterdir() if f.suffix.lower() in EXT)
        if args.limit:
            files = files[: args.limit]
        probs = predict_dir(model, files, size)
        called_sugarcane = probs[:, pos_index] >= 0.5
        correct = called_sugarcane if label == "sugarcane" else ~called_sugarcane
        totals[label] = (int(correct.sum()), len(files))
        print(f"  {label:<16}{correct.mean() * 100:6.1f} % correct   ({correct.sum()}/{len(files)})")

    got = sum(v[0] for v in totals.values())
    n = sum(v[1] for v in totals.values())
    print(f"\n  overall         {got / n * 100:6.1f} %   ({got}/{n})")

    # The decisive test: maize, held out from the source dataset directly.
    corn = _resolve("corn")
    files = [f for f in corn.rglob("*") if f.suffix.lower() in EXT]
    trained_names = {f.name for f in (args.holdout / "not_sugarcane").iterdir()}
    import hashlib

    unseen = []
    for f in files:
        try:
            if f"{hashlib.sha256(f.read_bytes()).hexdigest()[:16]}{f.suffix.lower()}" not in trained_names:
                unseen.append(f)
        except Exception:
            pass
        if len(unseen) >= 400:
            break
    if unseen:
        probs = predict_dir(model, unseen, size)
        wrong = (probs[:, pos_index] >= 0.5)
        print(f"\nMAIZE - the hard case ({len(unseen)} images)")
        print(f"  correctly rejected: {(~wrong).mean() * 100:.1f} %")
        if wrong.any():
            print(f"  called sugarcane:   {wrong.sum()} images")


if __name__ == "__main__":
    main()
