# Credits

Several methods in Vectraudit are our own implementations of ideas other people published or
built. Nothing below is a runtime dependency: no code from these projects runs inside
Vectraudit, nothing is downloaded from them, and nothing is sent to them. Each module's
docstring names its source too.

| Method | Where | Source |
|---|---|---|
| Reciprocal rank fusion, weighted-score fusion and score normalisation; the grid of fusion settings | `core/fusion.py`, `pipelines/fusion_replay.py`, `pipelines/fusion_grid.py` | Cormack, Clarke and Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (SIGIR 2009); the fusion and comparison design of [ranx](https://github.com/AmenRa/ranx) (Bassani) |
| Paired randomization test, randomized Tukey HSD, Holm adjustment | `evaluation/significance.py` | Smucker, Allan and Carterette, "A Comparison of Statistical Significance Tests for Information Retrieval Evaluation" (CIKM 2007); Carterette, "Multiple Testing in Statistical Analysis of Systems-Based Information Retrieval Experiments" (TOIS 2012); Holm (1979); the testing workflow of ranx |
| Wilson score interval | `evaluation/significance.py` | Wilson (1927) |
| Claim-level checking of answers: import of claim entailment results, claims as evidence for review | `evaluation/ragchecker_import.py`, `pipelines/ragchecker_input.py`, `pipelines/claim_review.py` | [RAGChecker](https://github.com/amazon-science/RAGChecker) (Ru et al., NeurIPS 2024). Vectraudit converts its results; it never runs it |
| Corpus scan, poison suite, canary check and red-team import for retrieval-time injection | `evaluation/corpus_scan.py`, `pipelines/poison*.py`, `pipelines/canary.py`, `evaluation/promptfoo_redteam_import.py` | The RAG plugins of [promptfoo](https://github.com/promptfoo/promptfoo) (rag-poisoning, rag-document-exfiltration, indirect-prompt-injection); Greshake et al., "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection" (2023); Zou et al., "PoisonedRAG" (USENIX Security 2025). Vectraudit imports promptfoo's results; it never runs it |
| Filtered search and pass-rate benchmarks; capacity under concurrency; dataset import with published neighbours | `core/filtering.py`, `core/selectivity.py`, `evaluation/capacity.py`, `evaluation/vdbbench_import.py`, `evaluation/external_reference.py` | [VectorDBBench](https://github.com/zilliztech/VectorDBBench) (Zilliz): its filter-rate cases, concurrent-search measurements and dataset layout |
| Interactive 2D map of the corpus and questions | `core/atlas.py` (builds the input), `frontend/src/views/CorpusAtlas.tsx` | [embedding-atlas](https://github.com/apple/embedding-atlas) (Apple, MIT), bundled with its required mosaic peers; see the third-party notices |
| Coverage regions (k-means with k-means++ seeding, TF-IDF labels) | `core/coverage.py` | Lloyd (1982); Arthur and Vassilvitskii, "k-means++" (SODA 2007); Sparck Jones (1972) |
| TREC run and qrels export and import, and the metric parity checks | `evaluation/trec.py`, `evaluation/measure_names.py` | The formats of NIST's `trec_eval`; [ir-measures](https://github.com/terrierteam/ir_measures) (Macdonald and MacAvaney) and [pytrec_eval](https://github.com/cvangysel/pytrec_eval) (Van Gysel and de Rijke) as the interchange and parity references. pytrec_eval only generated the committed fixture values; it is not a dependency |
| Judge-corrected correct rate with a valid interval (PPI and PPI++), and the random review sample it needs | `evaluation/ppi.py`, `evaluation/review_sample.py`, `evaluation/answer_estimates.py` | Angelopoulos, Bates, Fannjiang, Jordan and Zrnic, "Prediction-powered inference" (Science, 2023); Angelopoulos, Duchi and Zrnic, "PPI++: Efficient prediction-powered inference" (2023); [ARES](https://github.com/stanford-futuredata/ARES) (Saad-Falcon, Khattab, Potts and Zaharia, NAACL 2024). Vectraudit combines judgments you already have; it never runs a judge |
| Principal component projection | `core/projection.py` | Pearson (1901); Hotelling (1933) |

Where Vectraudit's behaviour differs from the original (for example the coverage rule, the
pass-rate convention, or that results are exploratory evidence and never a release approval),
the module docs and `docs/DECISIONS.md` say so. The frontend's bundled packages and their
licences are listed in `src/vectraudit/THIRD_PARTY_NOTICES.txt`.
