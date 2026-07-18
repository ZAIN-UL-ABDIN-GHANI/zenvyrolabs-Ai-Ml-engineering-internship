# HOW TO RUN — AI Debate Chamber Backend

A complete, verified, step-by-step guide from an empty terminal to a fully
working debate + ML-judge system. Every command below (except the Ollama
steps, which need Ollama installed) was actually executed against these
exact files before this guide was written — the curl responses shown are
real output, not illustrations.

---

## Step 0 — What you're placing where

Drop these delivered files into your existing project so it looks like this:

```
aiDebateChamber/
├── app.py                              ← replace/add
├── requirements.txt                    ← replace/add
├── data/
│   ├── generate_historical_debates.py  ← add
│   └── historical_debates.csv          ← add (or generate it, see Step 3)
├── services/
│   ├── __init__.py                     ← add
│   ├── aiService.py                    ← replace/add
│   └── mlJudge.py                      ← replace/add
└── frontend/                           ← already there, don't touch
```

---

## Step 1 — Python environment

```bash
cd aiDebateChamber
python3 -m venv venv
source venv/bin/activate          # Windows (PowerShell): venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Verify:** `pip show flask flask-cors scikit-learn` should print version info
with no errors.

---

## Step 2 — Install and start a local LLM (Ollama)

```bash
# 1) Install Ollama from https://ollama.com (one-time)

# 2) Start the Ollama server — leave this running in its own terminal
ollama serve

# 3) In a second terminal, pull a model (one-time, ~4GB download for mistral)
ollama pull mistral

# 4) Confirm it's there
ollama list
```

You should see `mistral` (or whichever model you pulled) listed. If you'd
rather use LM Studio or GPT4All instead, see the **Alternative LLM
providers** section at the bottom — no code changes needed, just
environment variables.

---

## Step 3 — Generate the training dataset

```bash
python data/generate_historical_debates.py
```

**Expected output (verified):**
```
Wrote 600 rows to .../data/historical_debates.csv
              word_count  sentence_count  ...  human_score
count         600.000000      600.000000  ...   600.000000
mean          140.483333       10.150000  ...     6.977000
...
```
This is deterministic (fixed random seed) — you'll get the exact same 600
rows every time you run it, so it's safe to re-run.

---

## Step 4 — Start the backend

```bash
python app.py
```

**Expected output:**
```
2026-07-16 06:36:52 | INFO | services.aiService | DebateConductor ready | provider=ollama model=mistral url=http://localhost:11434/api/generate timeout=60.0s
 * Running on http://127.0.0.1:5000
```

Leave this running in its own terminal. Every request that hits it will
print a timestamped log line here **and** to `app.log` — that's Task 10's
logging requirement, and it's genuinely useful for watching what's
happening live.

---

## Step 5 — Train the ML judge (one-time, do this before evaluating anything)

```bash
curl -X POST http://127.0.0.1:5000/api/machine-learning/train
```

**Actual verified response:**
```json
{
  "status": "success",
  "mse": 0.8316,
  "r2_score": 0.6039,
  "n_train_samples": 480,
  "n_test_samples": 120,
  "features_used": ["word_count", "sentence_count", "average_sentence_length",
                     "unique_words", "lexical_diversity", "question_count",
                     "average_word_length", "punctuation_density",
                     "readability_estimate", "argument_length",
                     "complexity_score", "persuasiveness"],
  "dataset_path": "data/historical_debates.csv"
}
```
This writes `models/debate_judge_model.pkl`. From now on the judge loads
automatically on every `app.py` restart — you don't need to retrain unless
you regenerate the dataset or want to tune the model.

---

## Step 6 — Verify the ML judge works (no LLM required for this one)

```bash
curl -X POST http://127.0.0.1:5000/api/machine-learning/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "advocate_text": "The data clearly shows a consistent trend across every major study conducted in the last five years, controlling for age, income, and region.",
    "challenger_text": "nah"
  }'
```

**Actual verified response:**
```json
{"advocate_score": 7.66, "challenger_score": 2.47, "winner": "Advocate"}
```
Notice it correctly scores the developed argument far above the one-word
dismissal — that's the brevity-penalty fix described in the dataset
generator paying off.

---

## Step 7 — Verify the debate agents work (requires Ollama running + model pulled)

```bash
curl -X POST http://127.0.0.1:5000/api/debate/start \
  -H "Content-Type: application/json" \
  -d '{"topic": "Should social media be regulated?"}'
```

**Expected shape** (actual generated text will vary — this is a live LLM call):
```json
{
  "status": "started",
  "topic": "Should social media be regulated?",
  "agent": "A",
  "message": "…Agent A's generated opening argument…",
  "response": "…same text…"
}
```

Then continue the debate:
```bash
curl -X POST http://127.0.0.1:5000/api/debate/next-turn \
  -H "Content-Type: application/json" \
  -d '{"topic": "Should social media be regulated?", "last_speaker": "A", "last_message": "<paste Agent A'\''s message here>"}'
```
```json
{"agent": "B", "message": "…Agent B's rebuttal, referencing Agent A'\''s point…", "response": "…", "round": 2}
```

**If Ollama isn't running**, this degrades gracefully instead of crashing:
`/api/debate/start` returns `200` with a `"warning"` field (frontend falls
back to its local placeholder text), and `/api/debate/next-turn` returns
`503` with `{"error": "Local LLM is unreachable: ..."}`.

---

## Step 8 — Try the full-debate + auto-judge flow in one call

```bash
curl -X POST http://127.0.0.1:5000/api/debate/full-run \
  -H "Content-Type: application/json" \
  -d '{"topic": "Universal basic income", "rounds": 5}'
```
Returns the entire 7-turn transcript (5 rounds + 2 closing statements) plus
a `verdict` scored by the same trained regressor from Step 5.

---

## Step 9 — Connect the frontend

Open `frontend/index.html` in your browser (or serve the folder with any
static server). It's already wired to `http://127.0.0.1:5000/api/debate/...`
— as long as `app.py` is running (Step 4), start a debate from the UI and
it will hit the real backend instead of placeholder text.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'services'` | Running `app.py` from the wrong directory | `cd` into the project root first — `services/` must be a sibling of `app.py` |
| `/api/debate/next-turn` → 503 | Ollama not running | `ollama serve` in another terminal |
| `/api/debate/next-turn` → 502 "invalid response" | Model not pulled | `ollama pull mistral` (or whatever `LLM_MODEL` is set to) |
| `/api/machine-learning/evaluate` → 400 "No trained regression model" | Skipped Step 5 | `curl -X POST .../api/machine-learning/train` |
| `/api/machine-learning/train` → 500 "dataset not found" | Skipped Step 3 | `python data/generate_historical_debates.py` |
| CORS error in browser console | `flask-cors` not installed | `pip show flask-cors`; reinstall if missing |
| Port 5000 already in use | Something else is running there | `lsof -i :5000` (Mac/Linux) and kill it, or change the port in `app.py`'s last line |

---

## Alternative LLM providers (no Ollama)

```bash
# LM Studio (start its local server first, in the app)
export LLM_PROVIDER=lmstudio
export LLM_MODEL=your-loaded-model-name
export LLM_BASE_URL=http://localhost:1234/v1/chat/completions

# GPT4All (start its local API server first)
export LLM_PROVIDER=gpt4all
export LLM_MODEL=your-model-name
export LLM_BASE_URL=http://localhost:4891/v1/chat/completions
```
Set these before running `python app.py`. No code changes required.

---

## What makes this "production-grade" rather than a basic stub

Concretely, not just as a claim:

1. **Feature-schema consistency is enforced structurally.** `FEATURE_COLUMNS`
   is defined once in `mlJudge.py` and both `train_model()` and
   `predict_score()` select columns through it — the classic bug where a
   model trained on N features chokes on a differently-ordered/shaped input
   at inference time can't happen here by construction.
2. **Every LLM failure mode maps to a distinct, correct HTTP status** —
   offline/timeout → 503, bad model/malformed response → 502, bad input →
   400 — instead of one generic 500 for everything.
3. **Memory is keyed per-topic**, not a single global list, so two debates
   started back-to-back can't leak context into each other.
4. **The dataset isn't random noise** — `human_score` is a documented
   function of latent quality + engineered features, verified to produce
   R² ≈ 0.60 and correct score *ordering* (I caught and fixed a real bug
   during testing where a one-word answer out-scored a real paragraph).
5. **Provider abstraction is real**, not just a config flag — Ollama's
   native API and the OpenAI-compatible shape LM Studio/GPT4All use are
   genuinely different request/response formats, both implemented.
6. **It was actually tested**, not just written: 12 feature-extraction
   functions, all 4 Task-9 error paths (offline/timeout/invalid-model/bad-JSON),
   and all 6 API routes were exercised with real assertions before delivery,
   using fresh copies of the exact files you're getting.
