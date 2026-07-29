#!/bin/sh
# Run both the FastAPI backend (port 8000) and the Streamlit UI (port 8501)
# in a single container.
set -e

uvicorn api.app:app --host 0.0.0.0 --port 8000 &
streamlit run api/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
