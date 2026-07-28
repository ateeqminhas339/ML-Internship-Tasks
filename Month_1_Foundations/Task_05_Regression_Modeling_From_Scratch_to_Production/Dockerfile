# Task 05 - Regression Modeling: From Scratch to Production
# Container image for the FastAPI insurance-charges prediction API.

FROM python:3.11-slim

WORKDIR /app

# Install only what the API needs at runtime (smaller image than the full
# notebook/dev requirements.txt).
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy the serving apps and the pre-trained model artifact.
COPY api/app.py ./api/app.py
COPY api/streamlit_app.py ./api/streamlit_app.py
COPY api/model.joblib ./api/model.joblib
COPY api/start.sh ./api/start.sh
RUN chmod +x ./api/start.sh

# 8000 = FastAPI JSON API (/predict, /health, /docs)
# 8501 = Streamlit browser UI
EXPOSE 8000 8501

CMD ["./api/start.sh"]
