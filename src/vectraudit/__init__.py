"""vectraudit: see inside your FAISS index, debug retrieval, tune recall.

import vectraudit
viewer = vectraudit.launch(index, vectors=xb, metadata=chunks)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vectraudit")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+unknown"

# Imported after __version__: the server and CLI modules read it from this package.
from vectraudit.session import Session
from vectraudit.viewer import Viewer, launch

__all__ = ["Session", "Viewer", "__version__", "launch"]
