import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Load the dataset (replace 'sales_data.csv' with your dataset)
df = pd.read_csv('sales_data.csv', parse_dates=['date'], index_col='date')

# Visualize the sales data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['sales'], marker='o', linestyle='-')
plt.title('Monthly Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# ARIMA model parameters
order = (1, 1, 1)  # Example ARIMA parameters (p, d, q)
seasonal_order = (0, 0, 0, 0)  # Example seasonal ARIMA parameters (P, D, Q, S)

# Splitting data into train and test sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Fit ARIMA model
model = ARIMA(train['sales'], order=order)
model_fit = model.fit()

# Forecast sales for next quarter
forecast_steps = len(test)
forecast, stderr, conf_int = model_fit.forecast(steps=forecast_steps)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(test['sales'], forecast)
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# Plotting the forecasts
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['sales'], label='Training Data')
plt.plot(test.index, test['sales'], label='Test Data')
plt.plot(test.index, forecast, label='ARIMA Forecast')
plt.title('ARIMA Sales Forecasting')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Visualize forecasts using Plotly
fig = px.line(df, x=df.index, y='sales', title='ARIMA Sales Forecasting')
fig.add_scatter(x=test.index, y=forecast, mode='lines', name='ARIMA Forecast')
fig.show()

