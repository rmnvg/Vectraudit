# VectorDBBench and ann-benchmarks datasets: external exact references

[VectorDBBench](https://github.com/zilliztech/VectorDBBench) ships public vector datasets with
the **exact nearest neighbours** of their test queries already computed. Vectraudit can use
those neighbours as the ground truth for ANN recall, so a backend with no exact-search mode
(or a corpus too large to brute-force here) can still be scored, and results line up with the
benchmark's own.

This is **not relevance labelling**. The neighbours say which vectors are nearest, not which
documents answer a question. Quality metrics such as nDCG still need labelled questions
([users bring labels](retrieval-workbench.md)).

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

## ann-benchmarks datasets

[ann-benchmarks](https://github.com/erikbern/ann-benchmarks) publishes each dataset (for
example `glove-100-angular.hdf5` or `sift-128-euclidean.hdf5`) as one HDF5 file with the train
vectors, the test queries and each query's 100 exact neighbours. Reading it needs the optional
`[hdf5]` extra (h5py); without it the command says how to install it or convert the file.

```bash
vectraudit dataset import-annb sift-128-euclidean.hdf5 --out sift
vectraudit dataset import-annb glove-100-angular.hdf5 --out glove --normalize
```

- **Layout and distances**, read in ann-benchmarks' own source: attributes `distance`,
  `type`, `dimension`, `point_type`; datasets `train`, `test`, `neighbors`, `distances`.
  `euclidean` becomes `l2` and `angular` (1 − cosine) becomes `cosine`. `hamming` (bit
  vectors) and `jaccard` (sparse sets) have no Vectraudit distance and are refused, as are
  sparse files. The distance is read from the file; `--distance` is only a check.
- **Checked against the data:** for 20 seeded test queries, the published distances must
  equal the distances recomputed from the vectors under the file's metric, or the import is
  refused. A file whose recorded metric does not describe its neighbours is never guessed at.
- **Outputs** are those of `import-vdbbench`: `vectors.npy`, `ids.npy` (train row `i` has id
  `i`, the ids the neighbours use), `queries.npy`, `dataset.jsonl` and `reference.json`
  (`external:annb`, with the file's hash).
- **Angular files and FAISS:** an angular file's vectors are not unit length, and a FAISS
  index ranks only by L2 or inner product, so a FAISS index cannot be checked against a cosine
  reference as such. `--normalize` writes unit-length vectors and queries; the angular
  neighbours are unchanged, an inner-product index (`IndexFlatIP`) then ranks exactly as
  cosine does, and `record` accepts it against that reference. Backends with a cosine distance
  (Qdrant, pgvector, ...) need no `--normalize`.
- **Recall definition:** ann-benchmarks counts a returned neighbour as found when its distance
  is within the k-th true distance plus 1e-3, so near-ties count; Vectraudit matches neighbour
  ids. On data with many exact ties the two can differ slightly.

## Use the reference

```bash
vectraudit record --connection connection.json --dataset cohere/dataset.jsonl \
  --reference cohere/reference.json --out run.json
vectraudit sweep index.faiss --queries cohere/queries.npy --reference cohere/reference.json
```

`record` measures `ann_recall` against the published neighbours instead of asking the backend
for its own exact search, and records the reference's source and hashes in the run. It
refuses, naming the cause, when:

- the backend's distance differs from the reference's (an inner-product backend is accepted
  for a cosine reference only when the reference was imported with `--normalize`);
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
