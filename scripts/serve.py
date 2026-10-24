"""Start one application worker using the host's assigned port."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    port = int(os.environ.get("PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    # In-memory throttling assumes one worker. Put a shared limiter in front of replicas.
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=port, workers=1, proxy_headers=True,
        app_dir=str(Path(__file__).resolve().parents[1]),
    )


if __name__ == "__main__":
    main()
