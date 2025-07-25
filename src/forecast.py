import numpy as np
import streamlit as st
from scipy import stats
import statsmodels.tsa.arima.model as arima_model
import statsmodels.tsa.holtwinters as holtwinters
from typing import Tuple
import pandas as pd


def forecast_spending(data: pd.Series, years_to_forecast: int, model_type: str = 'ARIMA') -> Tuple[np.ndarray, np.ndarray]:
    """Forecast future spending using various time series models."""
    history = data.values.astype(float)

    try:
        if model_type == 'ARIMA':
            try:
                if len(history) < 10:
                    st.warning(
                        "Not enough historical data for reliable ARIMA forecasting. Using linear regression instead.")
                    model_type = 'LinearRegression'
                else:
                    model = arima_model.ARIMA(history, order=(1, 1, 0))
                    model_fit = model.fit()
                    forecast_result = model_fit.get_forecast(steps=years_to_forecast)
                    forecasted_values = forecast_result.predicted_mean
                    confidence_intervals = forecast_result.conf_int(alpha=0.05)
                    return forecasted_values, confidence_intervals
            except Exception as e:
                st.warning(f"ARIMA forecasting failed: {e}. Using linear regression instead.")
                model_type = 'LinearRegression'

        if model_type == 'ExponentialSmoothing':
            try:
                if len(history) < 8:
                    st.warning(
                        "Not enough historical data for reliable ExponentialSmoothing. Using linear regression instead.")
                    model_type = 'LinearRegression'
                else:
                    seasonal_periods = min(5, len(history) // 2)
                    if len(history) >= 10:
                        model = holtwinters.ExponentialSmoothing(history, trend='add', seasonal='add',
                                                                 seasonal_periods=seasonal_periods)
                    else:
                        model = holtwinters.ExponentialSmoothing(history, trend='add', seasonal=None)

                    model_fit = model.fit(optimized=True)
                    forecasted_values = model_fit.forecast(steps=years_to_forecast)

                    residuals = model_fit.resid
                    std_error = np.std(residuals)

                    lower_ci = forecasted_values - 1.96 * std_error
                    upper_ci = forecasted_values + 1.96 * std_error
                    confidence_intervals = np.column_stack((lower_ci, upper_ci))

                    return forecasted_values, confidence_intervals
            except Exception as e:
                st.warning(f"ExponentialSmoothing forecasting failed: {e}. Using linear regression instead.")
                model_type = 'LinearRegression'

        if model_type == 'LinearRegression':
            years = np.arange(len(history))
            model = np.polyfit(years, history, 1)
            forecasted_years = np.arange(len(history), len(history) + years_to_forecast)
            forecasted_values = np.polyval(model, forecasted_years)

            fitted_values = np.polyval(model, years)
            residuals = history - fitted_values

            n = len(history)
            if n > 2:
                se = np.sqrt(np.sum(residuals ** 2) / (n - 2))
                se_forecast = se * np.sqrt(
                    1 + 1 / n + (forecasted_years - np.mean(years)) ** 2 / np.sum((years - np.mean(years)) ** 2))
                t_value = stats.t.ppf(0.975, n - 2)

                lower_ci = forecasted_values - t_value * se_forecast
                upper_ci = forecasted_values + t_value * se_forecast
            else:
                lower_ci = forecasted_values * 0.8
                upper_ci = forecasted_values * 1.2

            confidence_intervals = np.column_stack((lower_ci, upper_ci))
            return forecasted_values, confidence_intervals

    except Exception as e:
        st.error(f"All forecasting methods failed: {e}. Using simple trend extrapolation.")
        if len(history) >= 2:
            avg_growth = (history[-1] / history[0]) ** (1 / (len(history) - 1)) - 1 if history[0] > 0 else 0.05
            forecasted_values = np.array([history[-1] * (1 + avg_growth) ** (i + 1) for i in range(years_to_forecast)])
        else:
            forecasted_values = np.array([history[-1] * (1.02) ** (i + 1) for i in range(years_to_forecast)])

        lower_ci = forecasted_values * 0.7
        upper_ci = forecasted_values * 1.3
        confidence_intervals = np.column_stack((lower_ci, upper_ci))

    return forecasted_values, confidence_intervals
