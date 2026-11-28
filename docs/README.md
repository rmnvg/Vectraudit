# Vectraudit documentation

## Start here

- [Quickstart](quickstart.md): a red and a green pull-request check on a public benchmark in 10 minutes
- [Your first release check in 15 minutes](first-check.md), on your own backend
- [Case study: a latency fix that quietly broke retrieval](case-study-nprobe.md)

## Release checks

- [Release checks](release-checks.md): dataset, connection and policy formats, metrics, calibration, exceptions, reports and run files
- [Diagnose a setup offline](doctor.md): `vectraudit doctor`
- [Evidence a reviewer can verify](evidence-bundle.md): `check --bundle` and `vectraudit verify`
- [Tuning a search setting on any backend](tuning.md): `tune`, `tune-compare`
- [Releasing and CI installs](RELEASING.md): the pinned install action and upgrades
- [Pull-request workflow example](../examples/github/retrieval-check.yml)

## Labels and data you already have

- [Build a labelled dataset locally](labelling.md): `vectraudit label`
- [Label audit](label-audit.md): labels likely to be wrong or missing
- [Bring what you have](integrations.md): BEIR, TREC, MTCB, Ragas, DeepEval, notebooks, ANN benchmarks
- [VectorDBBench and ann-benchmarks datasets](vectordbbench.md): external exact references
- [Production intake](production-intake.md): new questions and drift from production traces

## Whole pipelines

- [Retrieval workbench](retrieval-workbench.md): pipelines, stage evidence, traces, answer judgments, investigations
- [Source-to-context lineage](lineage.md): where a required passage was first lost between source and model context
- [Permission lifecycle](permission-lifecycle.md) and [poisoning checks](poisoning.md)
- [Rank fusion](fusion.md), [attribution](attribution.md) and the [decision queue](decision-queue.md)
- [Claim-level evidence](claim-evidence.md) and [judge-corrected estimates](judge-corrected-estimates.md)
- [Migration report](migration-report.md)

## Choosing a setup

- [Experiments](experiments.md): embedding models, chunking and indexes on one labelled set
- [Case study: choosing an embedding model and chunk size](case-study-experiment.md)
- [Projecting recall to the whole corpus](corpus-projection.md)
- [Filtered search](filtered-search.md), [corpus atlas](corpus-atlas.md) and [capacity](capacity.md)

## FAISS

- [FAISS index debugger](faiss-debugger.md): views, supported indexes, LangChain and LlamaIndex folders, command line
- [Python and notebooks](jupyter.md)
- [Performance and memory](performance.md)

## Self-hosting and security

- [Security and data handling](security.md)
- [Running the control plane](cloud-operations.md) and [deployment](deployment.md)
- [Threat model](threat-model.md) and [incident response](incident-response.md)
- [Release rehearsal](../examples/release_demo.py) and the [reference support search](../examples/reference_search/README.md) examples

## Project

- [Decisions log](DECISIONS.md): one line per non-obvious technical decision
- [Credits](credits.md): the projects whose methods Vectraudit reimplements
- [Contributing](../CONTRIBUTING.md)
