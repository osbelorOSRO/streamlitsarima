"""
Implementación básica de pronóstico de series temporales
Nota: Esta es una versión simplificada sin bibliotecas externas.
Para uso en producción, es preferible utilizar statsmodels, pandas y numpy.
"""

import math
from datetime import datetime, timedelta

class SimpleTimeSeries:
    def __init__(self, data):
        self.data = data
        self.seasonal_period = 12  # Default monthly seasonality
    
    def calculate_moving_average(self, window_size=3):
        """Calculate simple moving average"""
        result = []
        for i in range(len(self.data) - window_size + 1):
            window = self.data[i:i + window_size]
            avg = sum(window) / window_size
            result.append(avg)
        return result
    
    def calculate_seasonal_factors(self):
        """Calculate basic seasonal factors"""
        seasonal_averages = [0] * self.seasonal_period
        seasonal_counts = [0] * self.seasonal_period
        
        for i, value in enumerate(self.data):
            season_index = i % self.seasonal_period
            seasonal_averages[season_index] += value
            seasonal_counts[season_index] += 1
        
        seasonal_factors = [
            avg / count if count > 0 else 1 
            for avg, count in zip(seasonal_averages, seasonal_counts)
        ]
        
        # Normalize factors
        factor_mean = sum(seasonal_factors) / len(seasonal_factors)
        seasonal_factors = [f / factor_mean for f in seasonal_factors]
        
        return seasonal_factors
    
    def forecast_next_periods(self, periods=12):
        """
        Simple forecasting using:
        - Moving average for trend
        - Seasonal factors for seasonality
        """
        if len(self.data) < self.seasonal_period * 2:
            raise ValueError("Not enough historical data for forecasting")
        
        # Calculate trend using moving average
        ma = self.calculate_moving_average(self.seasonal_period)
        
        # Calculate average change in trend
        trend_change = (ma[-1] - ma[0]) / len(ma)
        
        # Get seasonal factors
        seasonal_factors = self.calculate_seasonal_factors()
        
        # Last known level
        last_level = sum(self.data[-self.seasonal_period:]) / self.seasonal_period
        
        # Generate forecast
        forecast = []
        for i in range(periods):
            # Trend component
            trend = last_level + trend_change * (i + 1)
            # Seasonal component
            season_index = (len(self.data) + i) % self.seasonal_period
            seasonal = seasonal_factors[season_index]
            # Combine components
            forecast.append(trend * seasonal)
        
        return forecast

def main():
    # Example usage
    # Historical monthly sales data
    historical_data = [
        100, 120, 140, 160, 130, 110,  # First 6 months
        105, 125, 145, 165, 135, 115,  # Second 6 months
        110, 130, 150, 170, 140, 120   # Third 6 months
    ]
    
    # Create time series object
    ts = SimpleTimeSeries(historical_data)
    
    # Generate forecast for next 6 periods
    forecast = ts.forecast_next_periods(6)
    
    print("\nHistorical Data:")
    for i, value in enumerate(historical_data):
        print(f"Period {i + 1}: {value:.2f}")
    
    print("\nForecast:")
    for i, value in enumerate(forecast):
        print(f"Period {len(historical_data) + i + 1}: {value:.2f}")

if __name__ == "__main__":
    main()