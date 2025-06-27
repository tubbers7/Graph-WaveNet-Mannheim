import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima

def process_sensor(sensor_data_tuple):
    
    sensor_id, sensor_data = sensor_data_tuple

    try:
        print(f'Starting with sensor {sensor_id}')
        
        # Set the timestamp as index and keep only the 'traffic_flow' column
        sensor_data = sensor_data.set_index('timestamp')[['traffic_flow']]
        
        # Resample to ensure a consistent hourly frequency
        sensor_data = sensor_data.resample('h').mean()
        
        # Interpolate missing values
        traffic_flow = sensor_data['traffic_flow'].interpolate()
        
        # Split into train and test sets (e.g., last 20% for testing)
        train_size = int(len(traffic_flow) * 0.8)
        train, test = traffic_flow[:train_size], traffic_flow[train_size:]
    
        # Apply auto_arima to find the best ARIMA or SARIMA model
        model = auto_arima(
            train, 
            start_p=1, start_q=1,
            max_p=2, max_q=2, # Try with 2 on both
            d=1,
            seasonal=True,
            m=24, # Try with 24
            start_P=0, start_Q=0, 
            max_P=1, max_Q=1, 
            D=1,
            trace=False,
            error_action='ignore', 
            suppress_warnings=True,
            stepwise=True
        )
        
        # Forecast using the best model found by auto_arima
        forecast = model.predict(n_periods=len(test))

        # Calculate error metrics
        mae = mean_absolute_error(test, forecast)
        mape = np.mean(np.abs((test.values - forecast) / test.values)) * 100
        rmse = np.sqrt((test, forecast))

    except Exception as e:
        print(f"Error processing sensor {sensor_id}: {e}")
        return {'sensor_id': sensor_id, 'MAE': None, 'MAPE(%)': None, 'RMSE': None, 'Best_ARIMA_Order': None, 'Best_Seasonal_Order': None}
    
    
    
    # Return the result for this sensor
    return {
        'sensor_id': sensor_id,
        'MAE': mae,
        'MAPE(%)': mape,
        'RMSE': rmse,
        'Best_ARIMA_Order': model.order,
        'Best_Seasonal_Order': model.seasonal_order
    }

# Step 2: Run in parallel using Pool
if __name__ == "__main__":

    # Load data
    timeseries_data = pd.read_csv('/pfs/work9/workspace/scratch/ma_tofuchs-GraphWave-Seminar/Datasets/Mannheim/imputed_dataset_grin_full.csv', index_col=0)
    timeseries_data['timestamp'] = pd.to_datetime(timeseries_data['timestamp'])

    # Create a dataframe with only the first 10% of the data
    first_10_percent_data = timeseries_data.head(int(len(timeseries_data) * 0.1))

    sensor_data_groups = list(timeseries_data.groupby('sensor_id'))

    # Initialize a list to store results
    results = []

    # Process each sensor's data
    for sensor_data_tuple in sensor_data_groups:
        result = process_sensor(sensor_data_tuple)
        results.append(result)

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('auto-arima_results.csv', index=False)
    print("Processing complete. Results saved to auto-arima_results.csv")