"""Root FastAPI entrypoint — run the service directly with `python app.py`.

Usage:
    python app.py                  # serve on 0.0.0.0:8000
    python app.py --port 8001      # custom port
    python app.py --reload          # auto-reload for local dev
    uvicorn app:app --port 8000     # equivalent via uvicorn CLI

This is a thin wrapper over :mod:`triage_agent.api` so there is a single
source of truth for routes (`triage_agent/api.py`). It only adds:
  1. ``app`` re-export (so ``uvicorn app:app`` works from the repo root),
  2. ``.env`` loading via ``python-dotenv``,
  3. CLI flags (``--host/--port/--reload``) defaulting to ``HOST/PORT`` env.
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from triage_agent.api import app  # noqa: E402  (re-exported for `uvicorn app:app`)

__all__ = ["app", "main"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Support Ticket Triage API")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--reload", action="store_true", default=os.getenv("RELOAD", "").lower() in {"1", "true", "yes"})
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = _parse_args(argv)
    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
