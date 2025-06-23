# AIG 200 - Machine Learning Model Deployment

## Project Title: Bank Marketing Subscription Prediction

### Objective
The primary goal of this project was to develop a machine learning model to predict whether a client will subscribe to a bank's term deposit and to deploy this model as a scalable, secure REST API on a cloud platform.

### Repository Structure
This repository contains the following key files:
-   `Bank_Marketing_NN_Project.ipynb`: The Jupyter Notebook used for initial data exploration, visualization, and model experimentation.
-   `train.py`: A Python script to preprocess the data, train the final neural network model, and save the necessary artifacts.
-   `main.py`: The FastAPI application script that loads the model and serves predictions via a REST API.
-   `Dockerfile`: The recipe for building the containerized application.
-   `/artifacts`: The folder containing the trained model (`.keras`) and preprocessors (`.joblib`).
-   `requirements.txt`: The required Python libraries for the project.

### Setup and Running the Project Locally

**1. Environment Setup**
Adopted Conda environment on local machine
```bash
# Create and activate the environment
conda create -n aig200_deployment python=3.10
conda activate aig200_deployment

# Install dependencies
pip install -r requirements.txt
```

**2. Train the Model**
To retrain the model and generate the artifacts with the training script:
```bash
python train.py
```

**3. Running the API Locally with Docker**
```bash
# Build the Docker image
docker build -t bank-marketing-api .

# Run the Docker container
docker run -p 8080:8080 --name bank-marketing-container bank-marketing-api
```
The API is accessible at `http://localhost:8080`.

### API Usage

**1. Interactive Documentation (Swagger UI):**
The interactive API documentation is available at:
`http://localhost:8080/docs`

**2. Prediction Endpoint:**
-   **URL:** `/predict`
-   **Method:** `POST`
-   **Header:** `x-api-key: jy_key_aig200capstone`

**3. Example `curl` Request:**
```bash
curl -X 'POST' \
  'https://bank-marketing-service-573813129886.us-central1.run.app/predict' \
  -H 'accept: application/json' \
  -H 'x-api-key: jy_key_aig200capstone' \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 56,
    "job": "housemaid",
    "marital": "married",
    "education": "basic.4y",
    "housing": "no",
    "loan": "no",
    "contact": "telephone",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp_var_rate": 1.1,
    "cons_price_idx": 93.994,
    "cons_conf_idx": -36.4,
    "euribor3m": 4.857,
    "nr_employed": 5191.0
  }'
```

### Cloud Deployment & API Usage

The application was deployed to **Google Cloud Run**.

-   **Live API Endpoint:** `https://bank-marketing-service-573813129886.us-central1.run.app/predict`
-   **Interactive API Documentation (for browser-based testing):** `https://bank-marketing-service-573813129886.us-central1.run.app/docs#/default/predict_predict_post`
-   **API Key:** `jy_key_aig200capstone`