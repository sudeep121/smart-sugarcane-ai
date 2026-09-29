# Smart Sugarcane AI — running the complete project in VS Code

**Live public deployment:** https://sugarcane-irrigation.vercel.app — runs the
same trained models (soil, disease and the image validator) on ONNX Runtime.
Irrigation there falls back to the FAO-56 rule engine, because scikit-learn and
SciPy do not fit inside a serverless function; running locally gives you the
trained irrigation model as well.

This package is self-contained. It holds the source, the **trained models**, and the
**image datasets** the models were trained on. A plain clone of the GitHub repository
excludes the models and datasets (they are 676 MB), so a clone runs in demo mode —
this package does not.

```
SmartSugarcaneAI_Complete/
├── backend/        FastAPI service, models loader, validation gate, tests
├── frontend/       React 19 + TypeScript + Vite interface
├── ml/             Training and evaluation scripts
├── models/         Trained models (58 MB) — the app is in demo mode without these
├── datasets/       Images the models were trained on (618 MB)
├── data/           Agronomy reference data (varieties, fertilizer rules, soil profiles)
└── docs/           Architecture, API, dataset and deployment notes
```

## Requirements

| Tool | Version | Check with |
|---|---|---|
| Python | 3.12 (3.10–3.12 fine) | `python --version` |
| Node.js | 18 or newer | `node --version` |

Windows note: install Python from python.org and tick **"Add python.exe to PATH"**.

## First-time setup

Open the folder in VS Code, then open a terminal (`` Ctrl+` ``).

### 1. Backend

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

This installs ONNX Runtime, which is what loads the three image models. It is a
small download - TensorFlow is no longer needed to *run* the application, only to
retrain it. Serving on ONNX rather than TensorFlow cut memory from 579 MB to
117 MB, which is what allows the app to be hosted on a free tier at all.

### 2. Frontend

In a **second** terminal:

```powershell
cd frontend
npm install
```

## Running it

Two terminals, both left running.

**Terminal 1 — backend**

```powershell
cd backend
venv\Scripts\activate
python -m uvicorn app.main:app --reload --port 8000
```

Startup should report:

```
Irrigation : TRAINED MODEL
Disease    : TRAINED MODEL
Soil       : TRAINED MODEL
```

If any line says `DEMO` instead, the `models/` folder is missing or onnxruntime
did not install — see Troubleshooting.

**Terminal 2 — frontend**

```powershell
cd frontend
npm run dev
```

Open **http://localhost:5173**. Register an account (stored in a local SQLite file)
and the whole application is available.

To reach it from a phone on the same Wi-Fi, start the backend with
`--host 0.0.0.0`, allow port 8000 through Windows Firewall, and use the laptop's
`ipconfig` address. The login screen also has a **Server address** field, so the
address can be corrected inside the app rather than rebuilt.

## Verifying it works

```powershell
cd backend
venv\Scripts\activate
python smoke_test.py           # 84 checks: auth, analysis, history, image gate
python calibration_check.py    # 12 synthetic images through the demo heuristics
```

`smoke_test.py` creates a temporary account and deletes it afterwards.

## Retraining the models

The models in `models/` are already trained; these commands reproduce them.
Each takes 20–40 minutes on a CPU-only laptop.

```powershell
cd ml\disease_detection
pip install -r requirements.txt

# Disease classifier - 5 sugarcane conditions
python train.py --data-dir ..\..\datasets\sugarcane --epochs 15 --fine-tune-epochs 8

# Sugarcane image validator - is this photo sugarcane at all?
python train.py --data-dir ..\..\datasets\sugarcane_validator\train ^
  --output ..\..\models\sugarcane_validator_model.keras ^
  --meta ..\..\models\validator_model_meta.json ^
  --image-size 160 --epochs 10 --fine-tune-epochs 5

# Soil classifier - alluvial / black / clay / red
cd ..\soil_analysis
python train.py --data-dir ..\..\datasets\soil\clean --image-size 192 ^
  --epochs 20 --fine-tune-epochs 12
```

Evaluate on images training never saw:

```powershell
python ml\soil_analysis\evaluate.py                  # soil, 100 held-out images
python ml\disease_detection\evaluate_validator.py    # validator, 792 held-out + maize
```

`datasets/soil/holdout` and `datasets/sugarcane_validator/holdout` are quarantined
before training and must stay out of any training run — they are what make the
reported accuracies meaningful.

## What the numbers are

| Model | Type | Measured |
|---|---|---|
| Sugarcane image validator | MobileNetV2, 160×160 | 99.4 % on 792 held-out; **maize rejected 92.8 %** |
| Plant disease classifier | MobileNetV2, 224×224 | 86.3 % validation, 5 classes |
| Soil classifier | MobileNetV2, 192×192 | 93.0 % on 100 held-out |
| Irrigation regressor | Random Forest | R² 0.962, MAE 1.35 mm |
| Irrigation classifier | Random Forest | 96.4 % accuracy, F1 0.942 |
| FAO-56 water balance | Rule engine | not learned — agronomy |

Maize at 92.8 % is the honest weak figure: roughly 7 in 100 maize photographs are
still accepted as sugarcane. Quote that one rather than the 99.4 %.

## Troubleshooting

**Backend says DEMO instead of TRAINED MODEL**
`models/` is missing, or onnxruntime failed to install. Confirm with
`python -c "import onnxruntime; print(onnxruntime.__version__)"` inside the activated
venv. Python 3.10-3.12 are the tested versions.

**`venv\Scripts\activate` is blocked**
PowerShell execution policy. Run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, or
use `venv\Scripts\activate.bat` from cmd.

**"Cannot reach the backend"**
The backend terminal is not running, or it is on a different port. Both must be
running at once, which is why the instructions use two terminals.

**Port 8000 already in use**
An earlier backend is still alive. `netstat -ano | findstr :8000`, then
`taskkill /PID <pid> /F`.

**`npm install` fails on a corporate network**
`npm config set registry https://registry.npmjs.org/` and retry.

## An honest note on scope

The image models classify **appearance**. A photograph cannot measure nitrogen,
phosphorus, potassium, pH or electrical conductivity, and the application never
claims otherwise — every soil result says so and points to a laboratory test.
Every AI response carries a `model_source` field (`trained_model`, `rule_engine`
or `demo_heuristic`) that the interface renders as a visible badge, so a
heuristic estimate is never presented as a trained-model result.
