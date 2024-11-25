from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from Assignment2 import HousePricePredictor

# Initialize FastAPI app
app = FastAPI()

# Root endpoint to test the API
@app.get("/")
async def root():
    return {"message": "Welcome to the House Price Prediction API"}

# Pydantic model to define the input format
class HouseFeatures(BaseModel):
    MSZoning: str
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBsmtSF: float
    FullBath: int
    YearBuilt: int

# Instantiate the predictor
predictor = HousePricePredictor()
predictor.train()

# Predict endpoint to make predictions based on input features
@app.post("/predict")
async def predict_price(features: HouseFeatures):
    # Convert input features to a DataFrame
    input_data = pd.DataFrame([features.dict()])

    # Get the predicted price
    prediction = predictor.predict(input_data)

    # Return the prediction as a JSON response
    return {"predicted_price": float(prediction[0])}
