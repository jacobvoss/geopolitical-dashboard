import types
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import importlib


def load_forecast_spending():
    """Dynamically load forecast_spending from app.py without executing the app."""
    source_path = Path(__file__).resolve().parents[1] / "app.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    func_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "forecast_spending":
            func_node = node
            break
    assert func_node is not None, "forecast_spending not found"
    func_code = ast.get_source_segment(source, func_node)

    globals_dict = {
        "np": np,
        "arima_model": importlib.import_module("statsmodels.tsa.arima.model"),
        "holtwinters": importlib.import_module("statsmodels.tsa.holtwinters"),
        "stats": importlib.import_module("scipy.stats"),
        "st": types.SimpleNamespace(warning=lambda *a, **k: None, error=lambda *a, **k: None),
    }
    exec(func_code, globals_dict)
    return globals_dict["forecast_spending"]


forecast_spending = load_forecast_spending()


def test_forecast_spending_arima_runs_and_length():
    data = pd.Series(np.linspace(100, 150, 15))
    years = 5
    forecast, ci = forecast_spending(data, years, model_type="ARIMA")
    assert isinstance(forecast, np.ndarray)
    assert isinstance(ci, np.ndarray)
    assert len(forecast) == years
    assert ci.shape == (years, 2)


def test_forecast_spending_linear_regression_runs_and_length():
    data = pd.Series(np.linspace(100, 150, 15))
    years = 4
    forecast, ci = forecast_spending(data, years, model_type="LinearRegression")
    assert isinstance(forecast, np.ndarray)
    assert isinstance(ci, np.ndarray)
    assert len(forecast) == years
    assert ci.shape == (years, 2)
