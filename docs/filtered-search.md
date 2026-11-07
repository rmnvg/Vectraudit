# Filtered search

A metadata filter ("language is English", "tenant is acme") changes what a vector index can
retrieve. Vectraudit applies it the way a filtering index does, **inside the search**, and
measures recall against the exact neighbours **among the allowed ids**, never against the
unfiltered truth.

## How it works

- `core.filtering.allowed_from_metadata(metadata, filters, ids)` evaluates the filter once
  with the same portable matcher the release checks use (equality, list membership, numeric
  ranges) and returns an `AllowedSet`: the sorted user-facing ids that satisfy it. A document
  with no metadata row, or without the field, never satisfies a condition.
- `search(li, query, k, allowed=...)` hands FAISS an `IDSelector` through the search
  parameters. The index never returns a disallowed id. Fewer than `k` results come back
  (padded with `-1`) when the probed cells or visited nodes hold too few allowed vectors.
  That shortfall is the effect being measured, so it is not hidden.
- `GroundTruth.search(..., allowed=...)` runs exact k-NN over the allowed subset. The subset
  is cached by the set's fingerprint (the ids, not how the filter was written), two at a time.
- `explain_query(..., allowed=...)` explains a filtered IVF query with measured facts only:
  which true neighbours sat in unprobed cells, and how many allowed vectors the probed cells
  held (`allowed_in_probed`) against how many exist (`allowed_total`).

## Pass rate

`pass_rate` is the fraction of the corpus that **satisfies** the filter. Benchmarks such as
VectorDBBench quote a *filter rate*, the fraction **removed**. A 99% filter rate is a 1% pass
rate; convert explicitly when importing.

## Notes

- With a refine stage (`IVF..,RFlat`) FAISS honours the selector only inside the base
  parameters; the search code sets it there.
- A flat index under any filter has recall 1.0 by construction, which makes it a check on the
  ground truth itself.
