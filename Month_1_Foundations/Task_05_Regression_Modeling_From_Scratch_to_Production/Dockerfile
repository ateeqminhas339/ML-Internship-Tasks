# Task 05 - Regression Modeling: From Scratch to Production
# Container image for the FastAPI insurance-charges prediction API.

FROM python:3.11-slim

WORKDIR /app

# Install only what the API needs at runtime (smaller image than the full
# notebook/dev requirements.txt).
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy the serving app and the pre-trained model artifact.
COPY api/app.py ./api/app.py
COPY api/model.joblib ./api/model.joblib

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
