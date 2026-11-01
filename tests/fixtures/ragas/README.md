# Ragas results from a real run

Tables written by Ragas itself, not by hand: `evaluate(...).to_pandas()` saved with
pandas, so the importer is tested against what Ragas and pandas actually write. All text
is synthetic.

| File | Run | Saved with |
|---|---|---|
| `results.csv` | 4 samples; `ExactMatch`, `StringPresence`, `NonLLMStringSimilarity`, `BleuScore`, `RougeScore`, `NonLLMContextRecall` (no LLM) | `to_csv(index=False)` |
| `results.jsonl` | the same | `to_json(orient="records", lines=True)` |
| `results.json` | the same | `to_json()` (pandas' default column layout) |
| `failed.csv`, `failed.jsonl` | 2 samples; `ExactMatch` and `Faithfulness` with an unreachable judge, `raise_exceptions=False` | as above |

Versions: `ragas` 0.4.3, `pandas` 2.x, `langchain-openai` 1.7.3, Python 3.12.

Observed: the ROUGE column is named `rouge_score(mode=fmeasure)`; BLEU can exceed 1 by
rounding (`1.0000000000000004`); a failed metric is an empty cell in CSV and `null` in JSON.

To regenerate (in a separate environment; Ragas is not a Vectraudit dependency):

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install ragas rapidfuzz sacrebleu rouge_score \
  pandas "langchain-community<0.4"
RAGAS_DO_NOT_TRACK=true .venv/bin/python generate/run_ragas.py .
RAGAS_DO_NOT_TRACK=true .venv/bin/python generate/run_ragas_fail.py .
```

`langchain-community` 0.4 removed a module Ragas 0.4.3 imports, hence the pin.
