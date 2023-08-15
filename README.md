# Disease Prediction System

Ranking likely conditions from reported symptoms. A Random Forest trained on a
public symptom-disease dataset, served over FastAPI.

Work in progress.

## Planned

- [ ] Symptom to disease classifier
- [ ] FastAPI service around the trained model
- [ ] Log predictions to PostgreSQL to review what it gets wrong

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```
