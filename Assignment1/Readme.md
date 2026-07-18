# AI Debate Chamber — Backend Implementation

This document covers the backend pieces implemented for this assignment:
local-LLM-driven debate agents with memory (`services/aiService.py`), a
RandomForestRegressor-based scoring judge (`services/mlJudge.py`), the
synthetic training dataset, and the Flask API that wires them together
(`app.py`). The frontend (`frontend/`) was not modified.



---

## 1. Project Overview

Two LLM-backed personas debate a user-supplied topic over several rounds:

- **Agent A** — logical, evidence-driven, calm, refutes weak logic.
- **Agent B** — aggressive, persuasive, emotionally compelling, hunts for
  logical flaws (while staying respectful).

The backend keeps full per-topic conversation memory so every turn is
generated with the entire prior exchange as context, then a
`RandomForestRegressor`, trained on engineered linguistic features, scores
each side's argument(s) 1–10 and declares a winner.

## 2. Architecture

```
 ┌──────────────┐        JSON/HTTP         ┌──────────────────┐
 │  Frontend     │ ───────────────────────▶ │   Flask (app.py)  │
 │ (index.html,  │ ◀─────────────────────── │                    │
 │  app.js)      │                           └─────────┬─────────┘
 └──────────────┘                                      │
                                     ┌────────────────────┴───────────────────┐
                                     │                                        │
                          ┌──────────▼──────────┐                ┌────────────▼────────────┐
                          │ services/aiService.py │                │  services/mlJudge.py    │
                          │  DebateConductor       │                │  DebateRegressionJudge  │
                          │  - agent personas       │                │  - feature extraction   │
                          │  - per-topic memory      │                │  - RandomForestRegressor│
                          │  - debate engine          │                │  - train / predict      │
                          └──────────┬────────────┘                └────────────┬────────────┘
                                     │ HTTP                                     │ pandas / pickle
                          ┌──────────▼──────────┐                ┌────────────▼────────────┐
                          │ Local LLM              │                │ data/historical_debates │
                          │ Ollama (default)       │                │ .csv  →  models/*.pkl   │
                          │ or LM Studio / GPT4All │                └─────────────────────────┘
                          └────────────────────────┘
```

## 3. Folder Structure

```
project/
├── app.py                              # Flask routes (this file)
├── requirements.txt
├── data/
│   ├── historical_debates.csv          # synthetic training data (600 rows)
│   └── generate_historical_debates.py  # documented generator script
├── models/
│   └── debate_judge_model.pkl          # created after the first successful train
├── services/
│   ├── __init__.py
│   ├── aiService.py                    # DebateConductor (LLM agents + memory)
│   └── mlJudge.py                      # DebateRegressionJudge (features + model)
└── frontend/                           # unchanged
```

## 4. Installation

```bash
# from the project root
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Verified working versions (Python 3.12): Flask 3.1.3, Flask-Cors 6.0.5,
requests 2.33.1, pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0. Any
reasonably recent version of each should work on Python 3.10+; `pip` will
resolve compatible versions for your interpreter automatically since
`requirements.txt` is intentionally left unpinned for portability.

## 5. Running a Local LLM

**Ollama (default/preferred):**

```bash
# install from https://ollama.com, then:
ollama serve                 # starts the API on localhost:11434
ollama pull mistral          # or llama3 / phi3 / gemma3 / etc.
```

**LM Studio or GPT4All (alternative):** start the app's local server, then
set the environment variables below — no code changes needed:

```bash
export LLM_PROVIDER=lmstudio          # or gpt4all
export LLM_MODEL=your-local-model-name
export LLM_BASE_URL=http://localhost:1234/v1/chat/completions   # LM Studio default
```

| Variable              | Default                                    | Meaning                          |
|------------------------|---------------------------------------------|-----------------------------------|
| `LLM_PROVIDER`         | `ollama`                                    | `ollama` \| `lmstudio` \| `gpt4all` |
| `LLM_MODEL`            | `mistral`                                   | model name known to the provider |
| `LLM_BASE_URL`         | provider's default local port                | override the endpoint            |
| `LLM_REQUEST_TIMEOUT`  | `60`                                         | seconds before a generation call times out |

## 6. Running the Backend

```bash
python data/generate_historical_debates.py   # writes data/historical_debates.csv
python app.py                                 # starts Flask on :5000
```

On first run, call `POST /api/machine-learning/train` once (or use the demo
button if the frontend has one) to train and persist the regressor to
`models/debate_judge_model.pkl`; after that it loads automatically on
startup.

## 7. Running the Frontend

Open `frontend/index.html` directly, or serve the `frontend/` folder with
any static file server. It talks to `http://127.0.0.1:5000/api/debate/...`,
so the Flask server must be running first. CORS is enabled permissively in
`app.py` for local development.

---

## 8. API Documentation

All responses are JSON. Errors are always `{"error": "<message>"}` with a
non-200 status code.

### `POST /api/debate/start`
Starts a new debate session for a topic (resets memory for that topic) and
generates Agent A's opening statement.

```bash
curl -X POST http://127.0.0.1:5000/api/debate/start \
  -H "Content-Type: application/json" \
  -d '{"topic": "Should social media be regulated?"}'
```
```json
{
  "status": "started",
  "topic": "Should social media be regulated?",
  "agent": "A",
  "message": "…generated opening argument…",
  "response": "…generated opening argument…"
}
```

### `POST /api/debate/next-turn`
Generates the next agent's turn. The backend infers who should speak next
from `last_speaker` (A → B, anything else → A) and uses its own stored
memory for that topic as context — you don't need to resend the transcript.

```bash
curl -X POST http://127.0.0.1:5000/api/debate/next-turn \
  -H "Content-Type: application/json" \
  -d '{"topic": "Should social media be regulated?", "last_speaker": "A", "last_message": "…"}'
```
```json
{"agent": "B", "message": "…generated rebuttal…", "response": "…generated rebuttal…", "round": 2}
```

### `GET /api/debate/history` *(additive)*
```bash
curl "http://127.0.0.1:5000/api/debate/history?topic=Should%20social%20media%20be%20regulated%3F"
```
```json
{"topic": "…", "history": [{"agent": "A", "response": "…"}, {"agent": "B", "response": "…"}]}
```

### `POST /api/debate/full-run` *(additive)*
Runs an entire debate (opening → alternating rounds → closing statements)
in one call and scores it.

```bash
curl -X POST http://127.0.0.1:5000/api/debate/full-run \
  -H "Content-Type: application/json" \
  -d '{"topic": "Universal basic income", "rounds": 5}'
```
```json
{
  "topic": "Universal basic income",
  "rounds": 5,
  "transcript": [{"agent": "A", "response": "…"}, "…"],
  "verdict": {"advocate_score": 7.4, "challenger_score": 6.1, "winner": "Advocate"}
}
```

### `POST /api/machine-learning/train`
Trains the RandomForestRegressor on `data/historical_debates.csv` (80/20
split) and persists it to `models/debate_judge_model.pkl`.

```bash
curl -X POST http://127.0.0.1:5000/api/machine-learning/train
```
```json
{
  "status": "success", "mse": 0.8316, "r2_score": 0.6039,
  "n_train_samples": 480, "n_test_samples": 120,
  "features_used": ["word_count", "…"], "dataset_path": "data/historical_debates.csv"
}
```

### `POST /api/machine-learning/evaluate`
Scores an advocate's and a challenger's argument text and returns a winner.

```bash
curl -X POST http://127.0.0.1:5000/api/machine-learning/evaluate \
  -H "Content-Type: application/json" \
  -d '{"advocate_text": "…", "challenger_text": "…"}'
```
```json
{"advocate_score": 7.4, "challenger_score": 6.1, "winner": "Advocate"}
```

### Error examples

| Scenario                     | Status | Body                                                            |
|-------------------------------|--------|-------------------------------------------------------------------|
| Missing `topic`               | 400    | `{"error": "Field 'topic' is required and cannot be empty."}`     |
| Ollama not running             | 503    | `{"error": "Local LLM is unreachable: Could not reach ollama…"}`  |
| Unknown/unpulled model         | 502    | `{"error": "LLM returned an invalid response: Ollama returned HTTP 404…"}` |
| Dataset missing before train   | 500    | `{"error": "Historical dataset not found. Tried: […]…"}`          |
| Evaluate before any training   | 400    | `{"error": "No trained regression model is available yet…"}`      |
| Unknown route                  | 404    | `{"error": "Endpoint not found."}`                                 |

---

## 9. Memory System

`DebateConductor.debate_history` is a dict keyed by **topic**, each value an
ordered list of `{"agent": "A"|"B", "response": str}` turns. Every
generation call rebuilds the prompt from that list in the format:

```
Topic: <topic>

Previous Debate:
Agent A:
<...>

Agent B:
<...>

Now produce the next rebuttal.
```

Keying by topic (rather than one global list) means memory can't bleed
between unrelated debates, and starting a new debate on the same topic
(`/api/debate/start`) explicitly resets it via `reset_topic()`.

## 10. Regression Model

**Features** (`services/mlJudge.py`, `extract_features()` — every formula is
documented inline as a docstring on its extractor function):

| Feature | Formula |
|---|---|
| `word_count` | whitespace-token count |
| `sentence_count` | split on `.`/`!`/`?` |
| `average_sentence_length` | `word_count / sentence_count` |
| `unique_words` | distinct case-insensitive tokens |
| `lexical_diversity` | `unique_words / word_count` (type-token ratio) |
| `question_count` | count of `?` |
| `average_word_length` | mean characters per word |
| `punctuation_density` | punctuation chars / total chars |
| `readability_estimate` | Flesch Reading Ease approximation |
| `argument_length` | total character count |
| `complexity_score` | `0.4·avg_sentence_len + 1.5·avg_word_len + 5·lexical_diversity` |
| `persuasiveness` | density of persuasive/rhetorical markers, scaled 0–10 |

**Training:** `data/historical_debates.csv` (600 synthetic rows,
methodology documented at the top of `generate_historical_debates.py`) →
80/20 `train_test_split` → `RandomForestRegressor(n_estimators=200)` →
persisted with `pickle` to `models/debate_judge_model.pkl`. Current
verified metrics: **R² ≈ 0.60, MSE ≈ 0.83** on the held-out 20%.

**Why synthetic data:** no real corpus of debate transcripts with
human-assigned quality scores was available. The generator builds each row
from a hidden "quality" latent variable plus realistic feature ranges, and
computes `human_score` as a documented weighted function of that latent
quality, complexity, persuasiveness, and argument length — giving the
regressor genuine, non-trivial signal to learn rather than pure noise. This
is clearly synthetic data for exercising the ML pipeline end-to-end, not a
claim about real debate outcomes.

**Why RandomForest over Linear Regression:** the engineered features
interact non-linearly (e.g. complexity only helps up to a point, then
readability drops enough to hurt persuasiveness) — a tree ensemble
captures that without manual interaction terms, and is far more robust to
the outlier feature values short/degenerate arguments produce than a
linear model would be.

---

## 11. Testing

Every endpoint above was exercised with an automated test harness during
development (Flask `test_client`, mocked LLM HTTP layer, real ML judge) —
13 scenarios covering success paths, validation errors, LLM-offline,
invalid-model, invalid-JSON, and missing-dataset/model cases all pass. Use
the `curl` commands in section 8 to re-verify manually against your own
running server.

## 12. Troubleshooting

- **`Local LLM is unreachable` / connection refused** — Ollama isn't
  running or the wrong port is configured. Run `ollama serve` and confirm
  `ollama list` shows your model.
- **HTTP 502 "LLM returned an invalid response"** — usually means the
  configured `LLM_MODEL` hasn't been pulled. Run `ollama pull <model>`.
- **`/api/machine-learning/evaluate` returns 400 "No trained regression
  model"** — call `POST /api/machine-learning/train` once first.
- **`Historical dataset not found`** — run
  `python data/generate_historical_debates.py`.
- **CORS errors in the browser console** — confirm `app.py` is actually
  running on port 5000 and that `flask-cors` installed correctly
  (`pip show flask-cors`).

## 13. Future Improvements

- Swap the synthetic dataset for real, human-scored debate transcripts.
- Persist debate memory to disk/a database instead of an in-process dict,
  so history survives server restarts and scales beyond one process.
- Stream tokens from the LLM (`stream: true`) for a live-typing UI effect.
- Add authentication if this is ever exposed beyond localhost.
- Feature-importance endpoint exposing which linguistic features drove a
  given score (RandomForest makes this cheap via `.feature_importances_`).

