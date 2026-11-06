# RAGChecker's own example output

`checking_outputs.json` is `examples/checking_outputs.json` from
[amazon-science/RAGChecker](https://github.com/amazon-science/RAGChecker) (commit
`6091f08c00e6`, version 0.1.9), copied unchanged except for a final newline. RAGChecker is Apache-2.0
(`LICENSE`, `NOTICE` here); it is not a Vectraudit dependency. The file is the
`results` and `metrics` RAGChecker's evaluator wrote for two questions: claims, the
entailment labels between claims and retrieved chunks, and its own per-question metrics.

`tests/test_ragchecker_import.py` checks that the claim counts Vectraudit derives from the
labels reproduce all eleven of RAGChecker's per-question metrics, and that no claim,
response, reference answer or chunk text from this file reaches a judgment, a run or a
report. All text in the file is public (RAGChecker's own documentation example).
