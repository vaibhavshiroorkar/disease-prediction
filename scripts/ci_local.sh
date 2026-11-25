#!/usr/bin/env bash
# The CI steps from .github/workflows/ci.yml, run on this machine. Database
# tests skip unless TEST_DATABASE_URL is set. Set CI_DOCKER=1 for the image.
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin=.venv/Scripts/python
[ -x "$python_bin" ] || python_bin=.venv/bin/python

"$python_bin" -m ruff check app ml scripts study tests
node --check app/static/app.js
node --check app/static/lab.js
"$python_bin" -m pytest -ra
# BROWSER_CHANNEL=msedge uses installed Edge instead of downloaded Chromium.
RUN_BROWSER_TESTS=1 "$python_bin" -m pytest tests/browser -ra
if [ "${CI_DOCKER:-0}" = 1 ]; then
  docker build -t disease-prediction:test .
else
  echo "skipped: container build (set CI_DOCKER=1). CI on GitHub runs it."
fi
[ -n "${TEST_DATABASE_URL:-}" ] || echo "skipped: database tests. CI on GitHub runs them."
