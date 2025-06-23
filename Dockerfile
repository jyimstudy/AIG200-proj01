FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy artifacts into the container at /app
COPY . .

# Install needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define port 8080 for access outside this container
EXPOSE 8080


# use Uvicorn as ASGI server for FastAPI
# uses the $PORT environment variable provided by Cloud Run instead of hardcoded 8080
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
CMD uvicorn main:app --host 0.0.0.0 --port $PORT