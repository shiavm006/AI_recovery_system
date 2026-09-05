# Nakad
A failed-payment recovery agent for Indian merchants on Razorpay. It works out why each payment failed, spends a capped budget of mandate attempts and customer contacts on the ones worth it, refuses anything the rules disallow, and records every decision in a tamper-evident ledger.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas pyarrow pydantic python-dotenv pyyaml numpy requests \
  streamlit altair fastapi uvicorn razorpay chainladder google-genai openai
python -c "from generator.generate import freeze; freeze('data', seed=42)"
python run.py
python app.py && streamlit run app.py
uvicorn webhook:app --port 8000
```

Needs `NAKAD_LLM_PROVIDER` plus the matching API key (`GEMINI_API_KEY` / `OPENAI_API_KEY` / …); the webhook also needs `RAZORPAY_WEBHOOK_SECRET`.

## How it works

1. **Diagnose** — rule lookup, fleet outage correlation, then LLM on the rest.
2. **Allocate** — rank failures and spend capped retry and contact budgets.
3. **Govern** — refuse anything NPCI, RBI, Meta, or policy rules disallow.
4. **Simulate** — deterministic recovery outcomes for the batch (seeded coins).
5. **Ledger** — append ingest → diagnosis → proposal → gate as a hash chain.

Allocate, govern, and the ledger never call a model.

## Results

Against an arrival-order retry baseline, seed 42 recovers about **2.4×** as much (gross). Across five seeds the mean ratio is **2.59×** (sd **0.30**). The console artifact is `data/console.json`.

## What we know is wrong with it

- The lift is against an arrival-order baseline; against an amount-sorted one it is roughly 1.5×.
- The outcome simulator uses the same probability table the allocator ranks with, so the number depends on that assumption; under a fully inverted table it holds at 1.37×.
- Ranking by the index policy is close to equivalent to sorting by amount on this amount distribution.
- The LLM handles the ambiguous quarter of events and is worth a few percent of the lift.
- The chargeback reserve is fitted on all-method history but applied to card volume only.
- Outage correlation uses a 15-minute window, so gradual issuer degradation is not detected.

## Where the data comes from

Issuer mix is driven by NPCI published per-bank decline rates (`data/reference/npci-declines.csv`). Bank names come from the vendored [razorpay/ifsc](https://github.com/razorpay/ifsc) `banknames.json` map (MIT code; dataset public domain — see `data/reference/ifsc-banknames.LICENSE`). The batch is synthetic because real failure data has no labelled root cause, so nothing could be graded against it.
