"""Record a running Phoenix server's answers to phoenix-pull, for replay in tests.

Usage: python record.py http://127.0.0.1:6006 rag-demo SINCE UNTIL ../responses.json [small]

With ``small``, pages and ID batches are tiny so cursor paging and batching are recorded.
"""

import json
import sys
import urllib.request

from vectraudit.pipelines import phoenix_api
from vectraudit.pipelines.phoenix_api import PhoenixSource, fetch_annotations, fetch_spans

if sys.argv[6:] == ["small"]:
    phoenix_api.SPAN_PAGE, phoenix_api.ANNOTATION_PAGE, phoenix_api.ID_BATCH = 4, 2, 2

base, project, since, until, out = sys.argv[1:6]
recorded = {}


def send(url, headers, timeout):
    req = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    recorded[url[len(base) :]] = json.loads(body)
    return resp.status, body


source = PhoenixSource(base, send=send)
spans = fetch_spans(source, project, start=since, end=until)
fetch_annotations(source, project, spans)
with open(out, "w", encoding="utf-8") as f:
    json.dump(recorded, f, indent=1, sort_keys=True)
    f.write("\n")
print(len(recorded), "responses,", len(spans), "spans")
