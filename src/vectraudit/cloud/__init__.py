"""The hosted control plane: workspaces, tokens, uploaded runs, baselines, checks, reports.

Installed with ``vectraudit[cloud]`` and started with ``vectraudit-cloud serve``. It is a
separate service from the local FAISS UI (:mod:`vectraudit.server`). It never receives
user credentials: runners upload run files (ids, metrics, timings) only.
"""
