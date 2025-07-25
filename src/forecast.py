import numpy as np
from scipy import stats
import statsmodels.tsa.arima.model as arima_model
import statsmodels.tsa.holtwinters as holtwinters


def forecast_spending(data, years_to_forecast, model_type="ARIMA"):
    """Forecast future spending using ARIMA, ExponentialSmoothing or LinearRegression."""
    history = data.values.astype(float)

    try:
        if model_type == "ARIMA":
            if len(history) < 10:
                model_type = "LinearRegression"
            else:
                model = arima_model.ARIMA(history, order=(1, 1, 0))
                model_fit = model.fit()
                result = model_fit.get_forecast(steps=years_to_forecast)
                return result.predicted_mean, result.conf_int(alpha=0.05)

        if model_type == "ExponentialSmoothing":
            if len(history) < 8:
                model_type = "LinearRegression"
            else:
                seasonal_periods = min(5, len(history) // 2)
                if len(history) >= 10:
                    model = holtwinters.ExponentialSmoothing(
                        history, trend="add", seasonal="add", seasonal_periods=seasonal_periods
                    )
                else:
                    model = holtwinters.ExponentialSmoothing(history, trend="add", seasonal=None)
                model_fit = model.fit(optimized=True)
                forecasted = model_fit.forecast(steps=years_to_forecast)
                residuals = model_fit.resid
                std_error = np.std(residuals)
                lower_ci = forecasted - 1.96 * std_error
                upper_ci = forecasted + 1.96 * std_error
                return forecasted, np.column_stack((lower_ci, upper_ci))

        if model_type == "LinearRegression":
            years = np.arange(len(history))
            coeffs = np.polyfit(years, history, 1)
            forecast_years = np.arange(len(history), len(history) + years_to_forecast)
            forecasted = np.polyval(coeffs, forecast_years)
            fitted = np.polyval(coeffs, years)
            residuals = history - fitted
            n = len(history)
            if n > 2:
                se = np.sqrt(np.sum(residuals ** 2) / (n - 2))
                se_forecast = se * np.sqrt(1 + 1 / n + (forecast_years - np.mean(years)) ** 2 / np.sum((years - np.mean(years)) ** 2))
                t_value = stats.t.ppf(0.975, n - 2)
                lower_ci = forecasted - t_value * se_forecast
                upper_ci = forecasted + t_value * se_forecast
            else:
                lower_ci = forecasted * 0.8
                upper_ci = forecasted * 1.2
            return forecasted, np.column_stack((lower_ci, upper_ci))

    except Exception:
        pass

    if len(history) >= 2:
        avg_growth = (history[-1] / history[0]) ** (1 / (len(history) - 1)) - 1 if history[0] > 0 else 0.05
        forecasted = np.array([history[-1] * (1 + avg_growth) ** (i + 1) for i in range(years_to_forecast)])
    else:
        forecasted = np.array([history[-1] * 1.02 ** (i + 1) for i in range(years_to_forecast)])
    lower_ci = forecasted * 0.7
    upper_ci = forecasted * 1.3
    return forecasted, np.column_stack((lower_ci, upper_ci))
