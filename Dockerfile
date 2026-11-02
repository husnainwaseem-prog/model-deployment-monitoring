# Train the model at build time, then serve it with uvicorn.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

# Bake the trained model into the image (deterministic: fixed random_state).
RUN python -m src.cli train

EXPOSE 8000

CMD ["python", "-m", "src.cli", "serve"]
