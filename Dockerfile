# Employee Onboarding & Lifecycle Agent — service image
FROM python:3.12-slim

WORKDIR /app

# Dependencies first: this layer caches across code-only rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# .dockerignore keeps .env, reports, outbox and git history out of the image.
COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
