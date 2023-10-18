# model.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel
from tensorflow import keras

def load_models_from_folder(folder_name):
    # Get the path to the "weights" folder relative to the script location
    folder_path = os.path.join(os.path.dirname(__file__), folder_name)

    # Initialize an empty dictionary to store models
    models = {}

    # List all files in the specified folder
    model_files = os.listdir(folder_path)

    for file_name in model_files:
        if file_name.endswith(".h5"): 
            model_name = os.path.splitext(file_name)[0] 
            model_path = os.path.join(folder_path, file_name)
            model = keras.models.load_model(model_path)  
            models[model_name] = model 
    return models

# Define the path to the "weights" folder
weights_folder_path = "weights"

# Load models from the folder and store them in a dictionary
models = load_models_from_folder(weights_folder_path)

class CountryIndicator(BaseModel):
    country: str
    indicator: str
    start_date : str
    n_steps_future : int

def loading_model(country, indicator):
    model = None  # Initialize model as None
    
    for name, _ in models.items():
        if country in name and indicator in name:
            model = models[name]
            break  # Break out of the loop when a matching model is found
    
    return model  # Return the model, even if it's None


df = pd.read_csv('app/forecasting.csv',parse_dates=['Years'] , index_col='Years')
    
def forecasting(country, indicator, start_date, n_steps_future=2, data=df):
    # Filter data for the given country and indicator
    filtered_data = data[(data['Country'] == country) & (data['Indicator Name'] == indicator)]
    
    # Extract the 'Percentage' column values
    data_values = filtered_data['Percentage'].values
    
    # Load the model
    model = loading_model(country, indicator)
    
    # Generate forecast dates based on the specified start date and n_steps_future
    forecast_dates = pd.date_range(start=start_date, periods=n_steps_future, freq='A')

    # Generate forecasts for future time steps
    forecast = []
    last_sequence = data_values[-n_steps_future:] 

    for _ in range(n_steps_future):
        # Reshape the sequence for LSTM input
        sequence = last_sequence.reshape(1, -1, 1)

        # Make a prediction for the next time step
        predicted_value = model.predict(sequence)[0, 0]

        # Store the predicted value and update the last_sequence
        forecast.append(predicted_value)
        last_sequence = np.append(last_sequence[1:], predicted_value)

    # Create a Matplotlib plot for the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_dates, forecast, marker='o', linestyle='-', label=country)
    plt.title(f"Forecast for {indicator} in {country}")
    plt.xlabel("Date")
    plt.ylabel("Percentage")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

    # Return the forecasted values
    return forecast