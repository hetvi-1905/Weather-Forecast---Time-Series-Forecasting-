from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Initialize Flask App
app = Flask(__name__)

# Load weather data
data = pd.read_csv('weather_data.csv', parse_dates=['DATE'], index_col='DATE')

data = data.dropna()
# Split the dataset into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

@app.route('/', methods=['GET', 'POST'])
def index():
    forecast_result = None
    model_type = None
    mse_result = None

    if request.method == 'POST':
        # Get the selected model type from the form
        model_type = request.form['model']
        
        if model_type == 'ARIMA':
            # ARIMA model: fit to the avgtemp column
            model = ARIMA(train_data['AvgTemp'], order=(5, 1, 0))  # Adjust parameters as needed
            result = model.fit()

            # Forecast for the same length as the test data
            forecast = result.forecast(steps=len(test_data))
            mse_result = mean_squared_error(test_data['AvgTemp'], forecast)

        elif model_type == 'SARIMAX':
            # SARIMAX model: fit to the avgtemp column with seasonal order
            model = SARIMAX(train_data['AvgTemp'], 
                            order=(1, 1, 1), 
                            seasonal_order=(1, 1, 1, 12))  # Adjust parameters as needed
            result = model.fit()

            # Forecast for the same length as the test data
            forecast = result.forecast(steps=len(test_data))
            mse_result = mean_squared_error(test_data['AvgTemp'], forecast)

        # Convert forecasted values to a list for display
        forecast_result = forecast.tolist()

    # Render the index page with the forecast results
    return render_template('index.html', forecast=forecast_result, model=model_type, mse=mse_result)

if __name__ == '__main__':
    app.run(debug=True)
