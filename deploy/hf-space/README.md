---
title: vectraudit
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
license: other
short_description: See inside a FAISS index and debug why retrieval missed.
---

# vectraudit demo

A live [vectraudit](https://github.com/rmnvg/Vectraudit) instance on a real RAG corpus: 10,000
passage chunks embedded with `all-MiniLM-L6-v2`, indexed as `IVF256,PQ48x8` (inner product).

Try it:

- **Query explorer:** type a question (or pick a stored id) and see which true nearest
  neighbours the index missed and why: the cell wasn't probed, or PQ codes ranked it out.
- **Tuner:** sweep `nprobe` and find the cheapest setting for your target recall.
- **Quantization:** how much the 48-byte PQ codes distort distances.

This Space runs `vectraudit serve --demo-mode`: read-only, with capped `k`, sweep sizes and
`efSearch`, and UMAP served from a precomputed cache.

**Data:** "RAG Dataset 12000" by Neural Bridge AI
([neural-bridge/rag-dataset-12000](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000)),
Apache-2.0. Passages from Falcon RefinedWeb (ODC-By 1.0).

## Deploying this Space

Vectraudit is proprietary. Create the Space as **private** unless a public demo is a deliberate
launch decision; a public Space exposes the running app (not the Python source).

The Space repo needs only two files: this `README.md` and [`Dockerfile`](Dockerfile), both
copied from `deploy/hf-space/` in the Vectraudit repo. Add a Space secret named
`GITHUB_TOKEN`: a fine-grained GitHub token with read-only **Contents** access to
`rmnvg/Vectraudit` only. The image clones the repository at `VECTRAUDIT_REF` (default `main`;
pin a release tag for reproducible builds), builds the UI, bakes the demo data, model and
projections, and serves on port 7860. The token is mounted only during the clone; the `.git`
directory is removed so it never reaches an image layer.

To test the image locally from a Vectraudit checkout (no token needed):

```bash
docker build -f deploy/hf-space/Dockerfile -t vectraudit-space .
docker run --rm -p 7860:7860 vectraudit-space   # open http://localhost:7860
```
