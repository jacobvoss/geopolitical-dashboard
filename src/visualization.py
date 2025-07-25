import numpy as np
import plotly.graph_objects as go
import streamlit as st
from .forecast import forecast_spending


def create_spending_figure(df, country, compare_countries, data_source, forecast_years=0, forecast_model="ARIMA"):
    """Generate a plotly figure with optional forecast overlay."""
    if data_source == "SIPRI":
        fig = go.Figure()
        primary_data = df[df["Country"] == country]
        fig.add_trace(
            go.Scatter(
                x=primary_data["Year"],
                y=primary_data["Spending (USD)"],
                name=country,
                line=dict(color="#00d2d3", width=3),
                mode="lines+markers",
                hovertemplate="<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>",
            )
        )
        for c in compare_countries:
            comp_data = df[df["Country"] == c]
            fig.add_trace(
                go.Scatter(
                    x=comp_data["Year"],
                    y=comp_data["Spending (USD)"],
                    name=c,
                    line=dict(dash="dot", width=2),
                    mode="lines",
                    hovertemplate="<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>",
                )
            )
        y_title = "Spending (USD)"
        chart_title = f"{country} Military Spending — SIPRI"
    else:
        fig = go.Figure()
        primary_data = df[df["Country"] == country]
        fig.add_trace(
            go.Bar(
                x=primary_data["Year"],
                y=primary_data["Spending (% of GDP)"],
                name=country,
                marker_color="#00d2d3",
                hovertemplate="<b>%{x}</b><br>%{y:.2f}% of GDP<extra></extra>",
            )
        )
        for c in compare_countries:
            comp_data = df[df["Country"] == c]
            fig.add_trace(
                go.Bar(
                    x=comp_data["Year"],
                    y=comp_data["Spending (% of GDP)"],
                    name=c,
                    opacity=0.7,
                    hovertemplate="<b>%{x}</b><br>%{y:.2f}% of GDP<extra></extra>",
                )
            )
        years = primary_data["Year"]
        if not years.empty:
            min_year = min(years)
            max_year = max(years)
            fig.add_shape(
                type="line",
                x0=min_year,
                y0=2.0,
                x1=max_year,
                y1=2.0,
                line=dict(color="#ff6b4a", width=2, dash="dash"),
                name="NATO 2% Target",
            )
            fig.add_annotation(
                x=max_year,
                y=2.0,
                text="NATO 2% Target",
                showarrow=False,
                yshift=10,
                font=dict(color="#ff6b4a"),
                bgcolor="rgba(0,0,0,0.5)",
            )
        y_title = "Spending (% of GDP)"
        chart_title = f"{country} Defense Budget as % of GDP — NATO"

    fig.update_layout(
        title=chart_title,
        xaxis_title="Year",
        yaxis_title="Military Spending",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter"),
        hovermode="x unified",
        height=500,
        margin=dict(t=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="group",
    )

    if forecast_years > 0:
        last_year = df[df["Country"] == country]["Year"].max()
        historical = df[df["Country"] == country].set_index("Year")
        forecast_col = y_title
        if forecast_col in historical.columns:
            series = historical[forecast_col]
            forecasted_values, confidence = forecast_spending(series, forecast_years, forecast_model)
            years_list = list(range(last_year + 1, last_year + forecast_years + 1))
            fig.add_trace(
                go.Scatter(
                    x=years_list,
                    y=forecasted_values,
                    name=f"{forecast_model} Forecast",
                    line=dict(color="#ffdb58", dash="dash"),
                    mode="lines",
                    hovertemplate="<b>%{x}</b><br>%{y:.2f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=years_list + years_list[::-1],
                    y=np.concatenate([confidence[:, 1], confidence[:, 0][::-1]]),
                    fill="tozeroy",
                    fillcolor="rgba(255, 219, 88, 0.3)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="95% Confidence Interval",
                    hoverinfo="skip",
                )
            )
        else:
            st.error(f"The column '{forecast_col}' is not found in the data for {country}.")

    return fig, y_title
