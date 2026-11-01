"""Generate a real Ragas evaluate() result offline with non-LLM metrics."""

import sys
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    BleuScore,
    ExactMatch,
    NonLLMContextRecall,
    NonLLMStringSimilarity,
    RougeScore,
    StringPresence,
)

out = Path(sys.argv[1])
rows = [
    {
        "user_input": "How do I reset my password?",
        "response": "Open Settings and choose Reset password.",
        "reference": "Open Settings and choose Reset password.",
        "retrieved_contexts": ["Reset your password from Settings."],
        "reference_contexts": ["Reset your password from Settings."],
    },
    {
        "user_input": "Where is the invoice?",
        "response": "Invoices are emailed monthly.",
        "reference": "Invoices are under Billing > Invoices.",
        "retrieved_contexts": ["Invoices are emailed monthly."],
        "reference_contexts": ["Invoices are under Billing."],
    },
    {
        "user_input": "Cancel my plan",
        "response": "Go to Billing and press Cancel plan.",
        "reference": "Billing > Cancel plan.",
        "retrieved_contexts": ["Plans can be cancelled from Billing."],
        "reference_contexts": ["Plans can be cancelled from Billing."],
    },
    {
        "user_input": "What is the refund window?",
        "response": "30 days.",
        "reference": "Refunds are accepted within 30 days.",
        "retrieved_contexts": ["Refunds within 30 days."],
        "reference_contexts": ["Refunds within 30 days."],
    },
]
ds = EvaluationDataset.from_list(rows)
result = evaluate(
    ds,
    metrics=[
        ExactMatch(),
        StringPresence(),
        NonLLMStringSimilarity(),
        BleuScore(),
        RougeScore(),
        NonLLMContextRecall(),
    ],
    show_progress=False,
)
print(result)
df = result.to_pandas()
df.to_csv(out / "results.csv", index=False)
df.to_json(out / "results.jsonl", orient="records", lines=True)
df.to_json(out / "results.json")  # pandas default orient="columns"
print(list(df.columns))
