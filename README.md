# dependant-costs  —  U-CS 43

**Dependants Budget Management** · Wasaa Lifestyle

What supporting a child or relative costs over one, five and ten years, and which year is worst.

---

## Your contract

This is the shape the rest of the system expects. Agree any change with the teams below,
then tell Benjamin so the Team Integration Map gets reissued.

```
POST /projection
  in : { "dependantId": "..." }
  out: { "oneYear": 0, "fiveYear": 0, "tenYear": 0, "peakYear": 2036, "breakdown": {}, "assumptions": [] }
```

**Call it `dependant-costs` everywhere** — your repo name, your service URL, your route, and the
variable Benjamin sets when integrating you: `DEPENDANT_COSTS_SERVICE_URL`

## Who you depend on

U-CS 37 (budget-forecast) for the household cost model.

## Who depends on you

U-CS 46 (ops-metrics)

## Done when

Projections come with a stated range and assumptions, plus a note on handling children’s data.

---

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> — FastAPI gives you interactive API docs for free, so you can
test your endpoint without writing a client. Open `demo/index.html` for the mini app.

```bash
pytest -q                 # the contract test
docker build -t dependant-costs . && docker run -p 8000:8000 dependant-costs
```

## Progress

| Step | Status | Notes |
|------|--------|-------|
| **1 — Baseline** |  In progress | FastAPI scaffold is up; `service.py` still returns a stub — actual baseline computation not yet written |
| **2 — Evaluation** |  In progress | EDA started in `notebooks/aml.ipynb` and `notebooks/khbs2015_2016.ipynb`; metric and time-based split not yet finalised |
| **3 — Model** |  Not started | Waiting on baseline number |
| **4 — Integration & deploy** |  Not started | Waiting on model |

### What exists right now

- `app/main.py` — FastAPI app wired to `POST /projection` + `/health`
- `app/schemas.py` — Pydantic `Request`/`Response` contract shapes (fields still marked TODO)
- `app/service.py` — stub handler; **returns zeros** — baseline logic is next
- `data/aml.csv` — 836 rows of M-Pesa transaction data (date, category, amount\_in, amount\_out, balance)
- `notebooks/aml.ipynb` — exploratory analysis of the AML transaction data
- `notebooks/khbs2015_2016.ipynb` — exploratory analysis of KHBS 2015/16 household budget data
- `demo/index.html` — minimal front-end for manual testing
- `Dockerfile` + `.env.example` — deployment scaffold

### What you hand in

- This repo, on GitHub, with Benjamin added as a collaborator
- A live URL that answers
- The filled-in documents (templates are in the `documents/` folder of the skeleton repo)
- A README a stranger could run this from

## Rules

- No secrets in git. No `.env`, no keys. Commit `.env.example` instead.
- Label synthetic data as synthetic — in the code and in the report.
- Staging data only. Never production, never live M-Pesa.
- Tell Benjamin the same day if you find a real bug in the platform.
