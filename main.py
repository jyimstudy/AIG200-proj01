import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, Depends, HTTPException, status, Header
from pydantic import BaseModel
import tensorflow as tf

# --- Configuration ---
API_KEY = "jy_key_aig200capstone"

# --- API Key Authentication ---
def get_api_key(x_api_key: str = Header(...)):
    """Dependency function to verify the API key in the request header."""
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return x_api_key

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="An API to predict whether a client will subscribe to a term deposit.",
    version="1.0"
)

# --- Loading Model Artifacts ---
# Load the pre-trained model and preprocessing artifacts at startup.
try:
    model = tf.keras.models.load_model('bank_marketing_model.keras')
    target_encoder = joblib.load('target_encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load('model_columns.joblib')
    print("Model and artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading model artifacts: {e}")

# --- Pydantic Model for Input Data Validation ---
# Defines the data types and structure for the API request body.
class ClientData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the JY Bank Marketing Prediction API. Use the /predict endpoint for predictions."}


@app.post("/predict")
def predict(data: ClientData, api_key: str = Depends(get_api_key)):
    """
    Receives client data, preprocesses it, and returns a real-time prediction.
    """
    try:
        # 1. Convert incoming Pydantic data to a pandas DataFrame
        input_df = pd.DataFrame([data.dict()])

        # --- Rename columns to match the training format ---
        rename_map = {
            "emp_var_rate": "emp.var.rate",
            "cons_price_idx": "cons.price.idx",
            "cons_conf_idx": "cons.conf.idx",
            "nr_employed": "nr.employed",
            "day_of_week": "day_of_week", 
            "marital": "marital" 
        }
        # Rename only the columns that exist in the dataframe
        input_df.rename(columns={k:v for k,v in rename_map.items() if k in input_df.columns}, inplace=True)

        # 2. Apply the same preprocessing steps as in training
        input_df['contact'] = input_df['contact'].map({'telephone': 0, 'cellular': 1})
        label_map = {'no': 0, 'yes': 1, 'unknown': 2}
        input_df['housing'] = input_df['housing'].map(label_map)
        input_df['loan'] = input_df['loan'].map(label_map)
        edu_order = {'illiterate': 0, 'unknown': 1, 'basic.4y': 2, 'basic.6y': 3,
                     'basic.9y': 4, 'high.school': 5, 'professional.course': 6, 'university.degree': 7}
        input_df['education'] = input_df['education'].map(edu_order)
        month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        input_df['month'] = input_df['month'].map(month_map)

        input_df['previously_contacted'] = np.where(input_df['pdays'] != 999, 1, 0)
        input_df['was_contacted_before'] = np.where(input_df['previous'] > 0, 1, 0)
        input_df.drop(columns=['pdays', 'previous'], inplace=True)

        input_df['campaign'] = np.log1p(input_df['campaign'])

        input_df = pd.get_dummies(input_df, columns=['marital', 'day_of_week'], drop_first=True)

        input_df_aligned = input_df.reindex(columns=model_columns, fill_value=0)

        target_cols = ['job', 'poutcome']
        input_df_aligned[target_cols] = target_encoder.transform(input_df_aligned[target_cols])

        numeric_cols_with_dots = ['age', 'campaign', 'emp.var.rate', 'cons.price.idx',
                                'cons.conf.idx', 'euribor3m', 'nr.employed']
        input_df_aligned[numeric_cols_with_dots] = scaler.transform(input_df_aligned[numeric_cols_with_dots])

        # 3. Make Prediction
        prediction_proba = model.predict(input_df_aligned)[0][0]

        threshold = 0.58
        prediction = 1 if prediction_proba >= threshold else 0
        label = "yes" if prediction == 1 else "no"

        # 4. Return the result
        return {
            "prediction": label,
            "prediction_probability": float(prediction_proba)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )