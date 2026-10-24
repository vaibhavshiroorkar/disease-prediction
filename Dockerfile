FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    PRELOAD_MODELS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /srv
COPY requirements-runtime.txt ./
RUN python -m pip install --no-cache-dir --require-hashes -r requirements-runtime.txt \
    && useradd --create-home --uid 10001 app

COPY --chown=app:app app ./app
COPY --chown=app:app models ./models
COPY --chown=app:app scripts/serve.py ./scripts/serve.py
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/ready', timeout=4)"
CMD ["python", "scripts/serve.py"]
