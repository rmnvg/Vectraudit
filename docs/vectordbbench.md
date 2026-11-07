# VectorDBBench datasets and external exact references

[VectorDBBench](https://github.com/zilliztech/VectorDBBench) ships public vector datasets with
the **exact nearest neighbours** of their test queries already computed. Vectraudit can use
those neighbours as the ground truth for ANN recall, so a backend with no exact-search mode
(or a corpus too large to brute-force here) can still be scored, and results line up with the
benchmark's own.

This is **not relevance labelling**. The neighbours say which vectors are nearest, not which
documents answer a question. Quality metrics such as nDCG still need labelled questions
([customers bring labels](retrieval-workbench.md)).

## Get the data

Vectraudit never downloads datasets. Fetch one with VectorDBBench's own tooling, then point
Vectraudit at the directory. It expects the layout VectorDBBench uses:

| File | Holds |
|---|---|
| `train*.parquet` (or `shuffle_train*.parquet`) | `id`, `emb`: the corpus |
| `test.parquet` | `emb` (and optionally `id`): the queries |
| `neighbors.parquet` | `id` (a test query), `neighbors_id`: its exact neighbours |
| `neighbors_int_<p>p.parquet` (optional) | neighbours when the integer filter removes `<p>`% of the corpus |
| `scalar_labels.parquet` (optional) | `id`, `labels`: a label per vector |

The layout is read from VectorDBBench's documented format and tested against directories built
in that layout; it has not been checked against every published dataset.

## Import

```bash
pip install "vectraudit[parquet]"          # pyarrow
vectraudit dataset import-vdbbench ./cohere_small_100k --distance cosine --out ./cohere
```

`--distance` (`cosine`, `l2` or `ip`) is required and never guessed: the same vectors have
different nearest neighbours under different distances, and the reference is only valid for
the one it was computed under. The command writes:

- `vectors.npy` and `ids.npy`: the corpus, rows sorted by id (load them as `vectors` and `ids`
  in a FAISS connection, or load the same vectors into your backend under the same ids);
- `queries.npy` and `dataset.jsonl`: the test queries, with provenance (every source file's
  SHA-256) in the dataset header;
- `reference.json` + `reference.npy`: the published neighbours as an external exact reference
  (`reference_source: "external:vdbbench"`, `vectraudit.exact_reference` v1);
- for each `neighbors_int_<p>p.parquet`: `dataset-int-<p>p.jsonl`, `reference-int-<p>p.json`
  and a boolean metadata field `vdb_int_<p>p` in `metadata.jsonl`;
- `metadata.jsonl` also carries `label` when `scalar_labels.parquet` exists.

The source directory is only read.

### Filter rates and pass rates

VectorDBBench's **filter rate** is the fraction of the corpus *removed*; Vectraudit's **pass
rate** is the fraction that *passes*. The importer converts: `neighbors_int_99p` is a 1% pass
rate. VectorDBBench's integer filter keeps `id >= int(n * rate)`, and the importer checks that
every neighbour in the variant's file satisfies it. A file that does not is refused rather
than trusted.

To measure a filtered variant, load `metadata.jsonl` into your backend (or give the FAISS
connection `"metadata": "metadata.jsonl"`) and run the matching dataset:

```bash
vectraudit record --connection connection.json --dataset cohere/dataset-int-99p.jsonl \
  --reference cohere/reference-int-99p.json --out run.json
```

## Use the reference

```bash
vectraudit record --connection connection.json --dataset cohere/dataset.jsonl \
  --reference cohere/reference.json --out run.json
vectraudit sweep index.faiss --queries cohere/queries.npy --reference cohere/reference.json
```

`record` measures `ann_recall` against the published neighbours instead of asking the backend
for its own exact search, and records the reference's source and hashes in the run. It
refuses, naming the cause, when:

- the backend's distance differs from the reference's;
- `k` is larger than the number of neighbours the reference holds (whatever the file
  contains, often 100);
- the dataset has queries the reference does not cover, or carries a different filter from
  the one the reference assumes (neighbours for one filter say nothing about another);
- `--exact` is also given (two different exact references).

`sweep --reference` needs `--queries` to be the `queries.npy` written beside the reference
(checked by content), reports `truth_source: "external"`, and cannot be combined with
`--filter`/`--pass-rates` (use `record` with the filtered reference for that). The server and
the Tuner do not take a reference.

## Comparing runs

**Runs measured against different exact references are never compared on ANN recall.** The
backend's own exact search, VectorDBBench's neighbours and a different dataset's neighbours
answer different questions, so a policy rule on `ann_recall` is `not_comparable` (with the two
references named), and the comparison warns even when no rule needs it. Sweep runs saved with
different ground truth are likewise not compared on recall (`vectraudit runs diff` exits 4).
Re-record both sides against the same reference.
