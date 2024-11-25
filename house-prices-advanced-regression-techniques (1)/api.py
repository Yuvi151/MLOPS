'''Name:Yuvraj R.Jadhav
Roll No:391021
Prn:22210320
Batch:A1'''

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from Assignment2 import HousePricePredictor

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the House Price Prediction API"}


class HouseFeatures(BaseModel):
    MSZoning: str
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBsmtSF: float
    FullBath: int
    YearBuilt: int


predictor = HousePricePredictor()
predictor.train()


@app.post("/predict")
async def predict_price(features: HouseFeatures):
    input_data = pd.DataFrame([features.dict()])

    prediction = predictor.predict(input_data)

    return {"predicted_price": float(prediction[0])}
