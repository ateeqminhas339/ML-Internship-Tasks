FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching — only re-runs if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

EXPOSE 8888

# Default: launch Jupyter so you can open and run the notebook in a browser.
# --allow-root is needed because containers run as root by default.
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
