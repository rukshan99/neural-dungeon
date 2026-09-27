"""The browser face of the dungeon.

``dungeon serve`` starts a small local web server (FastAPI + uvicorn) that reads
the same manifests and save file as the CLI, streams trial runs into the page and
never becomes a second source of truth: every verdict still comes from pytest and
the ``dungeon.trials`` plugin.

Bound to 127.0.0.1 only. The server runs code from your clone; never expose it.
"""
