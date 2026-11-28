# Bring what you have, take results where you need them

Vectraudit reads the labels, runs and benchmarks you already have, and writes results in
formats other tools read. For production traces and answer judgments from observability and
evaluation tools, see the [workbench guide](retrieval-workbench.md#from-production-traces-and-the-tools-you-already-use).

- **Labels you already have:** `vectraudit dataset import` reads BEIR folders, TREC qrels,
  MTCB chunking benchmarks, Ragas test sets, DeepEval goldens and LangSmith/Langfuse exports
  ([formats](release-checks.md#bringing-labels-you-already-have)).
- **No labels yet:** the [local labelling assistant](labelling.md) (`vectraudit label`)
  pools candidate documents from your questions and corpus, then supports blind second
  passes, agreement and reviewer resolution before export. The
  [label audit](label-audit.md) later points reviewers at labels that look wrong.
- **Other IR tooling:** runs and labels move in and out in the TREC formats that `trec_eval`,
  pytrec_eval, ir-measures and ranx read (`runs export-trec`, `runs import-trec`,
  `dataset export-qrels`). An imported run says it is external and is never judged on latency
  it did not measure. Every quality metric is tested to equal trec_eval's, and reports show
  each metric's ir-measures name (`nDCG@10`, `RR@10`)
  ([details](release-checks.md#moving-runs-and-labels-to-and-from-other-ir-tools-trec)).
- **Your own notebooks:** `vectraudit runs table` writes one row per question and repeat
  (CSV or Parquet, no text), and `runs summary` validates a run of any size in a few MiB of
  memory ([large runs](release-checks.md#large-runs-summaries-and-tables)).
- **Public ANN benchmarks:** `dataset import-vdbbench` and `dataset import-annb` turn a
  VectorDBBench or ann-benchmarks dataset's published neighbours into the reference for ANN
  recall ([details](vectordbbench.md)).
- **Choosing a setup before you guard it:** [experiments](experiments.md)
  (`vectraudit experiment`) compare embedding models, chunkings and indexes on one labelled
  set, from your own corpus, a BEIR folder or an MTCB benchmark. Reports show MTCB's passage
  recall beside span recall, and `experiment export-mteb` writes each candidate in MTEB's
  result layout, marked as not like for like with leaderboard scores. `experiment project`
  projects each candidate's recall from a 10% sample to the whole corpus, with an interval
