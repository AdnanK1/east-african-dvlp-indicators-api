# main.py
from fastapi import FastAPI
from app.model import CountryIndicator, forecasting

app = FastAPI()

@app.get("/")
async def get_models():
    return {"Hello": "World"}


@app.post("/models/")
async def get_model(input: CountryIndicator):
    country = input.country
    indicator = input.indicator
    start_date = input.start_date
    n_steps_future = input.n_steps_future
    
    # Call the forecasting function and get the forecasted values
    forecast_values = forecasting(country, indicator, start_date, n_steps_future)
    
    if forecast_values is not None:
        return {"forecast": forecast_values}
    else:
        return {"error": "Model not found for the specified country and indicator"}