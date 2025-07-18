import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
from ml.data import process_data, apply_label
from ml.model import inference


# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


app = FastAPI()

# Load model components
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")


@app.get("/")
async def welcome():
    return {"message": "Welcome to the Income Prediction API!"}


@app.post("/data/")
async def predict(data: Data):
    try:
        # Convert data to DataFrame
        data_dict = data.dict(by_alias=True)
        data_df = pd.DataFrame([data_dict])

        # Process data
        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country"
        ]
        X, _, _, _ = process_data(
            data_df,
            categorical_features=cat_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb
        )

        # Predict and return
        pred = inference(model, X)
        #return {"prediction": apply_label(pred)}
        return "Hello from prediction function"
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))