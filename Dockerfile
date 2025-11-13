FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py /app/

# Expose the port the app will listen on inside the container
EXPOSE 8000

# Start the FastAPI app with Uvicorn. 
# Note: some hosts set $PORT; we handle both local (8000) and hosted ($PORT).
CMD ["bash", "-lc", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
