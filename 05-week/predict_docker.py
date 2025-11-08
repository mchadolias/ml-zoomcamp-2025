import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import uvicorn


class ClientData(BaseModel):
    lead_source: Literal[
        "paid_ads", "social_media", "events", "referral", "organic_search", "NA"
    ]
    number_of_courses_viewed: int = Field(ge=0)
    annual_income: float = Field(ge=0)


class PredictionResponse(BaseModel):
    converted_probability: float = Field(ge=0.0, le=1.0)
    converted: bool


app = FastAPI(title="Client Conversion Predictor")

with open("pipeline_v2.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)


def predict_single(client):
    result = pipeline.predict_proba(client)[0, 1]
    return float(result)


@app.post("/predict_docker")
def predict(client: ClientData) -> PredictionResponse:
    prob = predict_single(client.model_dump())
    return PredictionResponse(converted_probability=prob, converted=prob >= 0.5)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
