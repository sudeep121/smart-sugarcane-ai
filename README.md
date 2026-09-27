---
title: Smart Sugarcane AI
emoji: "🌱"
colorFrom: green
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Irrigation, disease and soil analysis for sugarcane
---

# Smart Sugarcane AI

AI-powered sugarcane irrigation, disease detection, soil analysis and fertilizer
recommendation system.

A full-stack web application: **FastAPI + SQLite + scikit-learn** backend,
**React + Vite + TypeScript + Tailwind** frontend. Runs entirely on your own
machine — no cloud account, no API keys required.

## Live deployment

**🌐 https://smart-sugarcane-ai.vercel.app**

Frontend and API are served from a single Vercel project (same origin, no CORS),
backed by hosted PostgreSQL.

Two differences from a local run, both deliberate and reported by the app itself
at `/api/system/status`:

| | Local | Vercel |
|---|---|---|
| Irrigation | **Trained Random Forest** | Rule engine |
| Uploaded photos | Stored and displayed | Analysed, not stored |

Vercel caps a Python function at 250 MB and the scientific stack is 457 MB, so
the deployment ships without scikit-learn and falls back to the water-balance
engine the model was trained to reproduce (R² 0.96 against it). Its filesystem
is also read-only, so photos are analysed in memory rather than saved. Run
`start.bat` locally for the full version.

---

## Table of contents

1. [What it does](#what-it-does)
2. [AI model status — read this first](#ai-model-status--read-this-first)
3. [Requirements](#requirements)
4. [Quick start](#quick-start)
5. [Running it in VS Code](#running-it-in-vs-code)
6. [Environment variables](#environment-variables)
7. [Testing the app](#testing-the-app)
8. [Training the ML models](#training-the-ml-models)
9. [Project structure](#project-structure)
10. [Deployment](#deployment)
11. [Troubleshooting](#troubleshooting)

---

## What it does

| Module | What you give it | What you get back |
|---|---|---|
| **Smart Irrigation** | Soil moisture, temperature, humidity, rainfall, rain chance, wind, weather, soil type, growth stage | Irrigate yes/no, water in mm and litres, duration, priority (Low → Critical), the reasoning, water-saving tips |
| **Disease Detection** | A photo of a leaf, stem or plant | Likely condition, confidence, severity, symptoms, causes, and a full recovery plan |
| **Plant Health History** | Repeat photos over time | Improving / stable / deteriorating trend with a health-score chart |
| **Soil Analysis** | A photo of bare soil | Visual soil category, moisture and texture appearance, organic matter appearance — plus explicit limitations |
| **Variety Recommendation** | Soil, region, climate, water availability, season | Ranked sugarcane varieties with match scores, reasoning and cautions |
| **Fertilizer Guidance** | Growth stage, soil type, optional lab NPK and pH | Nutrient priorities per stage, split application schedule, timing advice |
| **Sugarcane Assistant** | A question in plain language | An answer built from your own saved analyses, citing which records it used |

---

## AI model status — read this first

This project is deliberately honest about which parts are real machine learning
and which are not. Every AI response carries a `model_source` field, and the UI
shows a badge on every result.

| Module | Status out of the box | What that means |
|---|---|---|
| **Irrigation** | ✅ **Trained model** (after you run one command) | `RandomForestRegressor` + `RandomForestClassifier`, trained on a **synthetic** dataset generated from the project's water-balance model. Falls back to the transparent rule engine if the model file is missing. |
| **Disease detection** | ⚠️ **DEMO heuristic** | No labelled sugarcane disease dataset ships with this project, so there is nothing honest to train on. A transparent colour/texture heuristic runs instead, and every result is labelled `DEMO`. |
| **Soil analysis** | ⚠️ **DEMO heuristic** | Same situation. An HSV colour + graininess estimate, labelled `DEMO visual estimate`. |
| **Assistant** | ℹ️ **Rule-based** | Keyword intent matching over your saved records and the JSON knowledge base. Not a large language model, and it says so. |

**The demo heuristics are not fake.** They measure real properties of your image
— colour coverage, lesion fragmentation, edge density, graininess — and respond
to what is actually in the photo. They are validated against synthetic reference
images by `backend/calibration_check.py`. But they are **not** trained on real
sugarcane photographs, they are **not** diagnostically validated, and the app
never presents them as a diagnosis.

To replace them with real models, see [Training the ML models](#training-the-ml-models).

Check live status at any time:

```bash
curl http://localhost:8000/api/system/status
```

---

## Requirements

| Tool | Version | Check with |
|---|---|---|
| **Python** | 3.10 – 3.12 (tested on 3.12.10) | `python --version` |
| **Node.js** | 18 or newer (tested on 24.18.1) | `node --version` |
| **npm** | 9 or newer (tested on 11.16.0) | `npm --version` |

TensorFlow is **optional** — only needed if you train the image models.

---

## Quick start

### Easiest: double-click `start.bat` (Windows)

Once the one-time install below is done, just double-click **`start.bat`** in the
project root. It opens two windows — backend and frontend — and the browser
opens automatically.

**Keep both windows open while using the app.** Closing one stops that server,
and the app will show *"Cannot reach the backend"*. To stop everything, close
both windows.

You can also run them individually: `run-backend.bat` and `run-frontend.bat`.

### Manual: two terminals

Terminal 1 for the backend, terminal 2 for the frontend.

### Terminal 1 — Backend

```bash
cd backend
```

```bash
python -m venv venv
```

Activate it — **Windows**:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

Install and configure:

```bash
pip install -r requirements.txt
```

```bash
copy .env.example .env
```

(macOS / Linux: `cp .env.example .env`)

Start the API:

```bash
uvicorn app.main:app --reload
```

Backend runs at **http://localhost:8000** — interactive docs at
**http://localhost:8000/docs**.

### Terminal 2 — Frontend

```bash
cd frontend
```

```bash
npm install
```

```bash
copy .env.example .env
```

(macOS / Linux: `cp .env.example .env`)

```bash
npm run dev
```

Frontend runs at **http://localhost:5173** and opens automatically.

### Optional — train the irrigation model

From the project root, with the backend venv active:

```bash
python ml/irrigation/train.py
```

Takes about 30 seconds. It generates a synthetic dataset, trains both models and
saves them to `models/irrigation_model.joblib`. Restart the backend (or call
`POST /api/irrigation/reload-model`) and the irrigation module switches from
`rule_engine` to `trained_model`.

---

## Running it in VS Code

1. **File → Open Folder** and choose `smart-sugarcane-ai`.
2. Install the recommended extensions when VS Code prompts you (Python, Pylance,
   Tailwind CSS IntelliSense, ESLint).
3. Select the Python interpreter: `Ctrl+Shift+P` → *Python: Select Interpreter* →
   choose `./backend/venv/Scripts/python.exe`.
4. Open two terminals with ``Ctrl+Shift+` `` and follow the Quick start above.

The repo ships with `.vscode/launch.json`, so you can also press **F5** and pick
**"Backend: FastAPI (uvicorn)"** to run the API under the debugger, or
**"Full stack"** to launch both at once.

### URLs to open

| URL | What it is |
|---|---|
| http://localhost:5173 | The web application |
| http://localhost:8000/docs | Swagger UI — try any endpoint interactively |
| http://localhost:8000/redoc | Alternative API documentation |
| http://localhost:8000/api/system/status | Which AI models are actually loaded |

---

## Environment variables

### `backend/.env`

```env
SECRET_KEY=change_this_to_a_secure_secret
DATABASE_URL=sqlite:///./smart_sugarcane.db
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
UPLOAD_DIR=uploads
MODEL_MODE=auto
ACCESS_TOKEN_EXPIRE_MINUTES=10080
JWT_ALGORITHM=HS256
OPENWEATHER_API_KEY=
DEFAULT_WEATHER_CITY=Belagavi,IN
MAX_UPLOAD_MB=10
```

| Variable | Notes |
|---|---|
| `SECRET_KEY` | Signs JWTs. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(48))"`. **The app works with the default, but change it before exposing the server to anyone else.** |
| `DATABASE_URL` | SQLite by default. For PostgreSQL: `postgresql+psycopg://user:pass@localhost:5432/smart_sugarcane` |
| `MODEL_MODE` | `auto` (use trained models if present), `demo` (force heuristics), `production` (trained only) |
| `OPENWEATHER_API_KEY` | **Optional.** Free key from [openweathermap.org/api](https://openweathermap.org/api). Leave blank and the weather panel says so honestly — every other feature works unchanged. |

### `frontend/.env`

```env
VITE_API_BASE_URL=http://127.0.0.1:8000/api
VITE_MEDIA_BASE_URL=
```

Use `127.0.0.1`, not `localhost`. On Windows `localhost` resolves to `::1`
(IPv6) first while uvicorn binds IPv4 only, so the browser's first connection
attempt is refused. See [Troubleshooting](#troubleshooting).

Leave `VITE_MEDIA_BASE_URL` blank in development — the Vite dev server proxies
`/uploads` to the backend automatically.

Nothing is hard-coded: both `.env.example` files are committed, the real `.env`
files are gitignored.

---

## Testing the app

### Automated backend test

75 checks covering registration, login, JWT protection, every prediction
endpoint, upload validation, history and cleanup:

```bash
cd backend
```

```bash
venv\Scripts\python smoke_test.py
```

(macOS / Linux: `./venv/bin/python smoke_test.py`)

### Demo heuristic calibration

Confirms the image heuristics respond to the right visual cues:

```bash
venv\Scripts\python calibration_check.py
```

### Manual walkthrough

**Flow A — Irrigation**

1. Register at http://localhost:5173/register.
2. Go to **Smart Irrigation**.
3. Drag soil moisture to about **26 %**, set temperature **37**, humidity **38**.
4. Press **Get AI Recommendation**.
5. You should see `YES`, `HIGH` priority, a water figure in mm, a duration, a
   contribution chart and a plain-language explanation.
6. The result is saved — check the **History** page.

No soil moisture sensor? Enter days since last irrigation and press
**Simulate** to estimate it from a water balance.

**Flow B — Plant disease**

1. Go to **Plant Health**.
2. Drag in a photo of a sugarcane leaf (any leaf photo will produce a result —
   in DEMO mode it responds to colour and texture).
3. Press **Analyze Plant**.
4. You get the condition, confidence, severity, symptoms, causes and the
   "How to improve this plant" recovery plan.
5. Upload a second photo later to see the health trend chart appear.

**Flow C — Soil**

1. Go to **Soil Analysis**.
2. Upload a photo of bare soil. Optionally fill in region, water availability
   and season.
3. Press **Analyze Soil**.
4. You get the visual soil estimate, then **recommended varieties** and
   **fertilizer guidance** derived from it, and an explicit limitations panel.

**Flow D — Assistant**

Go to **AI Assistant** and click *"When should I irrigate my sugarcane?"*. It
answers from your actual saved irrigation record and cites it in the sidebar.

### Testing the API directly

Open http://localhost:8000/docs, use **Authorize** with a token from
`POST /api/auth/login`, then try any endpoint. Or:

```bash
curl -X POST http://localhost:8000/api/irrigation/predict -H "Content-Type: application/json" -d "{\"soil_moisture\":26,\"temperature\":37,\"humidity\":38,\"soil_type\":\"black\",\"growth_stage\":\"grand_growth\"}"
```

---

## Training the ML models

### Irrigation (fast, no extra dependencies)

```bash
python ml/irrigation/train.py
```

Add `--data path/to/your_field_data.csv` to train on real logged field data
instead of the synthetic set. See [`ml/irrigation/README.md`](ml/irrigation/README.md).

### Disease detection and soil (requires TensorFlow)

```bash
pip install -r ml/disease_detection/requirements.txt
```

This is a ~250 MB download. On Windows it installs `tensorflow-cpu`.

```bash
python ml/disease_detection/train.py --data-dir path/to/dataset
```

```bash
python ml/soil_analysis/train.py --data-dir path/to/soil_dataset
```

Dataset layout, where to find images, and how many you need are documented in
[`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md).

After training, restart the backend. The DEMO badges disappear automatically and
`model_source` changes to `trained_model` — no code changes needed.

---

## Project structure

```
smart-sugarcane-ai/
├── backend/
│   ├── app/
│   │   ├── api/            auth, irrigation, plants, soil, recommendations,
│   │   │                   history, assistant, weather, dashboard, system
│   │   ├── core/           config, security (JWT + bcrypt), dependencies
│   │   ├── models/         SQLAlchemy ORM models
│   │   ├── schemas/        Pydantic request/response models
│   │   ├── services/       business logic
│   │   ├── ml/             irrigation engine, image features, model loaders
│   │   ├── database.py
│   │   └── main.py         FastAPI entry point
│   ├── smoke_test.py       75-check end-to-end test
│   ├── calibration_check.py
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── components/     ui kit, ImageDropzone, ModelBadge, ProtectedRoute
│   │   ├── pages/          Landing, Login, Register, Dashboard, Irrigation,
│   │   │                   PlantAnalysis, SoilAnalysis, Varieties, Fertilizer,
│   │   │                   Assistant, History, Settings, NotFound
│   │   ├── layouts/        PublicLayout, DashboardLayout
│   │   ├── services/       axios client and typed endpoint wrappers
│   │   ├── context/        AuthContext, ToastContext
│   │   ├── hooks/          useTheme
│   │   ├── types/          shared TypeScript types
│   │   └── utils/          formatting helpers
│   ├── package.json
│   ├── vite.config.ts
│   └── .env.example
│
├── ml/
│   ├── irrigation/         generate_sample_dataset, train, predict
│   ├── disease_detection/  model, train, predict, class_names, requirements
│   └── soil_analysis/      model, train, predict, class_names
│
├── data/                   EDITABLE knowledge base — no code changes needed
│   ├── sugarcane_varieties.json
│   ├── fertilizer_rules.json
│   ├── disease_recommendations.json
│   └── soil_profiles.json
│
├── models/                 trained artifacts land here (gitignored)
├── uploads/                user photos (gitignored)
├── docs/
│   ├── SETUP.md
│   ├── API.md
│   ├── ML_MODELS.md
│   ├── DATASET_GUIDE.md
│   └── PROJECT_ARCHITECTURE.md
├── README.md
└── .gitignore
```

### The knowledge base is yours to edit

Everything the app says about varieties, fertilizer and diseases comes from the
four JSON files in `data/`. Correct them for your district, add local varieties,
adjust the fertilizer rules — no Python changes required. Reload without
restarting:

```bash
curl -X POST http://localhost:8000/api/recommendations/reload-knowledge
```

---

## Deployment

**Frontend → Vercel. Backend → a container host (Render / Railway / Fly.io).**

The backend cannot run on Vercel: its dependencies total **457 MB** against
Vercel's 250 MB serverless limit, and Vercel's ephemeral filesystem would
destroy the SQLite database and uploaded photos between requests. A container
host has none of those problems.

The repo ships everything needed:

| File | Purpose |
|---|---|
| `frontend/vercel.json` | Vite build + SPA rewrite + asset caching |
| `Dockerfile` | Backend image (build from the **project root**) |
| `.dockerignore` | Keeps local state out of the image |
| `render.yaml` | One-click Render blueprint |

Two environment variables make or break it on Vercel:

```
VITE_API_BASE_URL   = https://<your-backend>/api
VITE_MEDIA_BASE_URL = https://<your-backend>
```

Miss the second and every uploaded photo 404s.

Full walkthrough, free-tier caveats and a post-deploy checklist:
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'app'`**
Run uvicorn from inside the `backend/` directory, not the project root.

**`'uvicorn' is not recognized`**
The virtual environment is not active. Run `venv\Scripts\activate` first — your
prompt should show `(venv)`.

**Frontend shows "Cannot reach the backend"**

*By far the most common cause: the backend window was closed, or was never
started.* The frontend and backend are two separate programs — running only
`npm run dev` gives you a working-looking UI that cannot load any data. Check
that the backend window is still open and shows `Application startup complete.`

Also give it a few seconds: the backend loads a 17 MB model file at startup, so
there is a short window after launch where the UI is up but the API is not
answering yet.

Then confirm it responds: http://127.0.0.1:8000/api/system/health

If that URL works in your browser but the app still cannot reach it, you have hit
the **IPv6 `localhost` trap**, which is the most common cause on Windows.
`localhost` resolves to `::1` (IPv6) before `127.0.0.1`, but uvicorn binds to
IPv4 only by default — so the browser's first attempt is refused. Note that
`curl` hides this by silently falling back to IPv4, so a working `curl` proves
nothing here.

Fix: make sure `frontend/.env` uses the IPv4 address, then **restart the Vite
dev server** (it only reads `.env` at startup):

```env
VITE_API_BASE_URL=http://127.0.0.1:8000/api
```

Alternatively, bind uvicorn to both stacks with `uvicorn app.main:app --reload --host ::`.

**CORS errors in the browser console**
Add your frontend origin to `CORS_ORIGINS` in `backend/.env`, comma-separated,
then restart the backend.

**Images upload but do not display**
In development, Vite proxies `/uploads` to port 8000 — make sure the backend is
running. In production, set `VITE_MEDIA_BASE_URL` to your backend's base URL.

**`bcrypt` or `passlib` errors**
This project uses `bcrypt` directly and does not depend on passlib, so the
common passlib/bcrypt version clash does not apply. If bcrypt itself fails to
install, upgrade pip: `python -m pip install --upgrade pip`.

**Disease detection always says DEMO**
That is correct and expected until you train a model — see
[Training the ML models](#training-the-ml-models). Confirm with
`GET /api/plants/model-status`.

**Port already in use**
Backend: `uvicorn app.main:app --reload --port 8001` (then update
`VITE_API_BASE_URL`). Frontend: `npm run dev -- --port 5174` (then add that
origin to `CORS_ORIGINS`).

If port 8000 stays occupied after you closed the window, uvicorn's `--reload`
worker can survive its parent and keep the socket open. Find and stop it:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

If the port is still held by a PID that no longer exists, the worker is a
`multiprocessing.spawn` child — list `python.exe` processes and stop the one
whose parent PID matches the dead process.

---

## Responsible use

This is a **decision-support tool**, not certified agricultural advice.

- Image analysis results are estimates, not diagnoses. Confirm any disease or
  chemical treatment decision with a qualified agricultural officer.
- A photograph cannot measure NPK, pH, electrical conductivity or
  micronutrients. Use a laboratory soil test for those.
- The fertilizer engine gives **priorities and timing**, never fixed dosages —
  safe quantities depend on your soil test, variety and state recommendation.
- Variety match scores rank knowledge-base entries against your stated
  conditions. They are not predicted yields.
