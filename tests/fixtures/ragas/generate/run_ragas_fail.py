"""A real Ragas run where an LLM metric fails (unreachable judge) next to a non-LLM one."""

import sys
from pathlib import Path

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ExactMatch, Faithfulness

out = Path(sys.argv[1])
llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="judge-model", base_url="http://127.0.0.1:9/v1", api_key="x", max_retries=0, timeout=2
    )
)
rows = [
    {
        "user_input": "How do I reset my password?",
        "response": "Open Settings and choose Reset password.",
        "reference": "Open Settings and choose Reset password.",
        "retrieved_contexts": ["Reset your password from Settings."],
    },
    {
        "user_input": "Where is the invoice?",
        "response": "Invoices are emailed monthly.",
        "reference": "Invoices are under Billing > Invoices.",
        "retrieved_contexts": ["Invoices are emailed monthly."],
    },
]
result = evaluate(
    EvaluationDataset.from_list(rows),
    metrics=[ExactMatch(), Faithfulness(llm=llm)],
    raise_exceptions=False,
    show_progress=False,
    run_config=RunConfig(max_retries=1, timeout=5),
)
df = result.to_pandas()
df.to_csv(out / "failed.csv", index=False)
df.to_json(out / "failed.jsonl", orient="records", lines=True)
