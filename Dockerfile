# Two stages: build the React bundle with Node, then serve everything from a
# slim Python image that never installs Node or PyTorch.

# --- stage 1: frontend -----------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /app
COPY dashboard/package.json dashboard/package-lock.json* ./
RUN npm install --no-fund --no-audit
COPY dashboard/ ./
RUN npm run build

# --- stage 2: runtime ------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code. src/fourthdown/models/ imports torch, but engine.py only
# touches it inside _load_torch(), which the numpy backend never calls.
COPY src/ ./src/
COPY api/ ./api/

# Inference artifacts: the exported numpy weights and the fitted scalers.
COPY models/weights.npz models/weights_meta.json ./models/
COPY models/preprocessor.json models/preprocessor.npz ./models/

# The three tables the dashboard reads. Training inputs are not copied.
COPY data/processed/scored_fourth_downs.parquet \
     data/processed/coach_summary.parquet \
     data/processed/season_summary.parquet ./data/processed/

COPY --from=frontend /app/dist ./dashboard/dist

EXPOSE 8000

# One worker: the engine holds the models and a 41k-row frame in memory, and
# extra workers duplicate both for no benefit at demo traffic. Threads handle
# concurrency instead. The grid endpoint is the slow one and it is memoized,
# so the timeout only needs to cover a cold call.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", \
     "--timeout", "60", "--access-logfile", "-", "api.app:app"]
