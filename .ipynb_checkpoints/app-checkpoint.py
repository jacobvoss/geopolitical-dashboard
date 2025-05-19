import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit.components.v1 import html
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

# ===== CONFIG =====
st.set_page_config(
    layout="wide",
    page_title="NATO Defence Analytics",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# ===== STYLES =====
def apply_styles():
    st.markdown(f"""
    <style>
    :root {{
        --primary: #00d2d3;
        --secondary: #ff6b4a;
        --bg: #0f172a;
        --card: #1e293b;
        --text: #e2e8f0;
        --positive: #00d2d3;
        --negative: #ff6b4a;
        --neutral: #94a3b8;
        --forecast: #ffd166;
    }}
    .metric-card {{
        background-color: var(--card);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }}
    
    .event-table {{
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }}
    
    .event-table th,
    .event-table td {{
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    
    .event-table th {{
        font-weight: 500;
        color: #94a3b8;
    }}
    
    .event-table th:nth-child(1) {{
        width: 15%;
    }}
    
    .event-table th:nth-child(2) {{
        width: 55%;
    }}
    
    .event-table th:nth-child(3) {{
        width: 30%;
        text-align: right;
    }}
    
    .event-table td:nth-child(3) {{
        text-align: right;
    }}
    
    .positive-change {{
        color: var(--positive);
    }}
    
    .negative-change {{
        color: var(--negative);
    }}
    
    .fade-in {{
        opacity: 1;
        transition: opacity 0.5s ease;
    }}
    
    /* Forecast options styling */
    .stRadio > div {{
        flex-direction: row;
        gap: 10px;
    }}
    
    .stRadio label {{
        background-color: var(--card);
        padding: 8px 16px;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 0;
        transition: all 0.2s ease;
    }}
    
    .stRadio label:hover {{
        border-color: var(--primary);
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

apply_styles()

# ===== DATA LOADING =====
@st.cache_data
def load_data(source="SIPRI"):
    if source == "SIPRI":
        df = pd.read_csv('cleaned_data/SIPRI_spending_clean.csv')
    else:
        df = pd.read_csv('cleaned_data/nato_defense_spending_clean.csv')
    
    df_melted = df.melt(id_vars=['Country'], var_name='Year', value_name='Spending')
    df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
    df_melted = df_melted.dropna(subset=['Year'])
    df_melted['Year'] = df_melted['Year'].astype(int)

    # Only calculate YoY changes for SIPRI (not for NATO)
    if source == "SIPRI":
        df_melted['YoY_Change'] = df_melted.groupby('Country')['Spending'].pct_change() * 100
        df_melted.rename(columns={'Spending': 'Spending (USD)'}, inplace=True)
    else:
        df_melted.rename(columns={'Spending': 'Spending (% of GDP)'}, inplace=True)
    
    return df_melted.dropna()

# ===== FORECAST FUNCTIONS =====
def generate_forecast(data, method="ARIMA", forecast_years=5, confidence=0.95):
    """Generate forecast based on historical data using various methods"""
    # Extract years and values
    years = data['Year'].values
    values = data['Spending'].values
    
    # Check if we have enough data
    if len(values) < 5:
        return None, None, None
    
    # Create forecast years
    last_year = years[-1]
    forecast_x = np.array([last_year + i + 1 for i in range(forecast_years)])
    
    # Different forecast methods
    if method == "Linear":
        # Simple linear regression
        coeffs = np.polyfit(years, values, 1)
        poly = np.poly1d(coeffs)
        forecast_y = poly(forecast_x)
        
        # Create confidence intervals
        residuals = values - poly(years)
        std_dev = np.std(residuals)
        z_value = 1.96  # 95% confidence interval
        
        conf_interval = z_value * std_dev
        upper_bound = forecast_y + conf_interval
        lower_bound = forecast_y - conf_interval
        
    elif method == "Exponential":
        # Use Holt-Winters exponential smoothing
        try:
            model = ExponentialSmoothing(
                values, 
                trend='add', 
                seasonal=None,
                initialization_method='estimated'
            ).fit()
            
            forecast_y = model.forecast(forecast_years)
            
            # Confidence intervals
            residuals = values - model.fittedvalues
            std_dev = np.std(residuals)
            z_value = 1.96
            
            conf_interval = z_value * std_dev * np.sqrt(np.arange(1, forecast_years + 1))
            upper_bound = forecast_y + conf_interval
            lower_bound = forecast_y - conf_interval
            
        except Exception as e:
            # Fall back to linear regression if exponential fails
            coeffs = np.polyfit(years, values, 1)
            poly = np.poly1d(coeffs)
            forecast_y = poly(forecast_x)
            
            residuals = values - poly(years)
            std_dev = np.std(residuals)
            z_value = 1.96
            
            conf_interval = z_value * std_dev
            upper_bound = forecast_y + conf_interval
            lower_bound = forecast_y - conf_interval
    
    elif method == "ARIMA":
        try:
            # Use ARIMA model (p=1, d=1, q=0) is a reasonable default
            model = ARIMA(values, order=(1, 1, 0))
            model_fit = model.fit()
            
            forecast_result = model_fit.get_forecast(steps=forecast_years)
            forecast_y = forecast_result.predicted_mean
            
            # Get confidence intervals
            conf_int = forecast_result.conf_int(alpha=1-confidence)
            lower_bound = conf_int.iloc[:, 0].values
            upper_bound = conf_int.iloc[:, 1].values
            
        except Exception as e:
            # Fall back to linear regression if ARIMA fails
            coeffs = np.polyfit(years, values, 1)
            poly = np.poly1d(coeffs)
            forecast_y = poly(forecast_x)
            
            residuals = values - poly(years)
            std_dev = np.std(residuals)
            z_value = 1.96
            
            conf_interval = z_value * std_dev
            upper_bound = forecast_y + conf_interval
            lower_bound = forecast_y - conf_interval
    
    else:  # Moving Average
        # Use 3-year moving average for forecasting
        window = min(3, len(values))
        last_values = values[-window:]
        avg = np.mean(last_values)
        
        # Simple trend calculation - average change over the last window years
        if len(values) > window:
            trend = (values[-1] - values[-window-1]) / window
        else:
            trend = 0
            
        forecast_y = np.array([avg + trend * i for i in range(1, forecast_years + 1)])
        
        # Confidence based on standard deviation of recent changes
        recent_changes = np.diff(values[-window-1:])
        std_dev = np.std(recent_changes) if len(recent_changes) > 0 else 0
        z_value = 1.96
        
        conf_intervals = [z_value * std_dev * np.sqrt(i) for i in range(1, forecast_years + 1)]
        upper_bound = forecast_y + conf_intervals
        lower_bound = forecast_y - conf_intervals
    
    # Ensure no negative values in lower bound for spending
    lower_bound = np.maximum(0, lower_bound)
    
    return forecast_x, forecast_y, (lower_bound, upper_bound)

# ===== EVENTS DATA =====
EVENTS = {
    "Global": {
        2001: "9/11 Attacks",
        2008: "Global Financial Crisis",
        2014: "Crimea Annexation",
        2020: "COVID-19 Pandemic",
        2022: "Russia Invades Ukraine",
        2023: "2023 Gaza War"
    }
}

def calculate_event_impact(country, year, df):
    is_nato = 'Spending (% of GDP)' in df.columns
    country_data = df[df['Country'] == country].sort_values('Year').reset_index(drop=True)
    
    try:
        event_idx = country_data.index[country_data['Year'] == year][0]
    except IndexError:
        return None
        
    if event_idx == 0:
        return None
    
    if is_nato:
        # For NATO: Absolute difference in percentage points
        current = country_data.at[event_idx, 'Spending (% of GDP)']
        previous = country_data.at[event_idx - 1, 'Spending (% of GDP)']
        change = current - previous
    else:
        # For SIPRI: Percentage change
        current = country_data.at[event_idx, 'Spending (USD)']
        previous = country_data.at[event_idx - 1, 'Spending (USD)']
        change = (current - previous) / previous * 100
    
    return {
        'year': year,
        'name': EVENTS.get("Global", {}).get(year),
        'change': change,
        'prev_year': country_data.at[event_idx - 1, 'Year'],
        'is_nato': is_nato
    }

# ===== UI =====
st.title("Defence Spending Analytics")
st.caption("NATO military expenditure trends with event impact analysis")

with st.sidebar:
    st.header("Filters")
    data_source = st.selectbox("Select Data Source", ["SIPRI", "NATO"])
    df = load_data(data_source)
    available_countries = df['Country'].unique().tolist()
    default_index = available_countries.index('United States') if 'United States' in available_countries else 0
    country = st.selectbox("Select Country", available_countries, index=default_index)
    compare_countries = st.multiselect("Compare With", [c for c in available_countries if c != country])
    
    st.header("Forecast Settings")
    show_forecast = st.checkbox("Show Forecast", value=True)
    if show_forecast:
        forecast_years = st.slider("Forecast Years", min_value=1, max_value=10, value=5)
        forecast_method = st.radio("Forecast Method", 
                                  ["ARIMA", "Linear", "Exponential", "Moving Average"], 
                                  horizontal=True)
        confidence_level = st.slider("Confidence Level", min_value=0.7, max_value=0.99, value=0.95, format="%.2f")

# Main layout
col1, col2 = st.columns([3, 1])

with col1:
    if data_source == "SIPRI":
        fig = go.Figure()
        primary_data = df[df['Country'] == country]
        
        # Sort data by year to ensure proper line plots
        primary_data = primary_data.sort_values('Year')
        
        fig.add_trace(go.Scatter(
            x=primary_data['Year'],
            y=primary_data['Spending (USD)'],
            name=country,
            line=dict(color='#00d2d3', width=3),
            mode='lines+markers',
            hovertemplate="<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>"
        ))
        
        # Add forecast if enabled
        if show_forecast and 'show_forecast' in locals():
            forecast_data = primary_data.rename(columns={'Year': 'Year', 'Spending (USD)': 'Spending'})
            forecast_x, forecast_y, confidence_intervals = generate_forecast(
                forecast_data, 
                method=forecast_method, 
                forecast_years=forecast_years, 
                confidence=confidence_level
            )
            
            if forecast_x is not None:
                # Add forecast line
                fig.add_trace(go.Scatter(
                    x=forecast_x,
                    y=forecast_y,
                    name="Forecast",
                    line=dict(color='#ffd166', width=3, dash='dash'),
                    mode='lines',
                    hovertemplate="<b>%{x}</b><br>$%{y:,.0f}M (Forecast)<extra></extra>"
                ))
                
                # Add confidence intervals
                lower_bound, upper_bound = confidence_intervals
                fig.add_trace(go.Scatter(
                    x=np.concatenate([forecast_x, forecast_x[::-1]]),
                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255, 209, 102, 0.2)',
                    line=dict(color='rgba(255, 209, 102, 0)'),
                    name=f"{int(confidence_level*100)}% Confidence",
                    hoverinfo="skip"
                ))

        for c in compare_countries:
            comp_data = df[df['Country'] == c].sort_values('Year')
            fig.add_trace(go.Scatter(
                x=comp_data['Year'],
                y=comp_data['Spending (USD)'],
                name=c,
                line=dict(dash='dot', width=2),
                mode='lines',
                hovertemplate="<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>"
            ))

        y_title = "Military Spending (USD Millions)"
        chart_title = f"{country} Military Spending — SIPRI"

    else:  # NATO Data
        fig = go.Figure()
        primary_data = df[df['Country'] == country].sort_values('Year')
        
        # For NATO data, we'll use lines instead of bars for better forecast visualization
        fig.add_trace(go.Scatter(
            x=primary_data['Year'],
            y=primary_data['Spending (% of GDP)'],
            name=f"{country}",
            line=dict(color='#00d2d3', width=3),
            mode='lines+markers',
            hovertemplate="<b>%{x}</b><br>%{y:.2f}% of GDP<extra></extra>"
        ))
        
        # Add forecast if enabled
        if show_forecast and 'show_forecast' in locals():
            forecast_data = primary_data.rename(columns={'Year': 'Year', 'Spending (% of GDP)': 'Spending'})
            forecast_x, forecast_y, confidence_intervals = generate_forecast(
                forecast_data, 
                method=forecast_method, 
                forecast_years=forecast_years, 
                confidence=confidence_level
            )
            
            if forecast_x is not None:
                # Add forecast line
                fig.add_trace(go.Scatter(
                    x=forecast_x,
                    y=forecast_y,
                    name="Forecast",
                    line=dict(color='#ffd166', width=3, dash='dash'),
                    mode='lines',
                    hovertemplate="<b>%{x}</b><br>%{y:.2f}% of GDP (Forecast)<extra></extra>"
                ))
                
                # Add confidence intervals
                lower_bound, upper_bound = confidence_intervals
                fig.add_trace(go.Scatter(
                    x=np.concatenate([forecast_x, forecast_x[::-1]]),
                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255, 209, 102, 0.2)',
                    line=dict(color='rgba(255, 209, 102, 0)'),
                    name=f"{int(confidence_level*100)}% Confidence",
                    hoverinfo="skip"
                ))
        
        for c in compare_countries:
            comp_data = df[df['Country'] == c].sort_values('Year')
            fig.add_trace(go.Scatter(
                x=comp_data['Year'],
                y=comp_data['Spending (% of GDP)'],
                name=c,
                line=dict(dash='dot', width=2),
                mode='lines',
                hovertemplate="<b>%{x}</b><br>%{y:.2f}% of GDP<extra></extra>"
            ))
        
        # Add 2% target line
        years = primary_data['Year']
        if not years.empty:
            min_year = min(years)
            max_year = max(years) + (forecast_years if show_forecast and 'show_forecast' in locals() else 0)
            
            fig.add_shape(
                type="line",
                x0=min_year,
                y0=2.0,
                x1=max_year,
                y1=2.0,
                line=dict(color="#ff6b4a", width=2, dash="dash"),
                name="NATO 2% Target"
            )
            
            fig.add_annotation(
                x=max_year,
                y=2.0,
                text="NATO 2% Target",
                showarrow=False,
                yshift=10,
                font=dict(color="#ff6b4a"),
                bgcolor="rgba(0,0,0,0.5)"
            )
        
        y_title = "Military Spending (% of GDP)"
        chart_title = f"{country} Defense Budget as % of GDP — NATO"

    fig.update_layout(
        title=chart_title,
        xaxis_title="Year",
        yaxis_title=y_title,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family="Inter"),
        hovermode="x unified",
        height=500,
        margin=dict(t=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Add forecast explanation if showing forecast
    if show_forecast and 'show_forecast' in locals() and forecast_x is not None:
        with st.expander("About this forecast"):
            st.markdown(f"""
            #### {forecast_method} Forecast Methodology
            
            This forecast projects defense spending for **{country}** over the next **{forecast_years} years** using a {forecast_method.lower()} model.
            
            - **Confidence Level**: {confidence_level*100:.0f}%
            - **Data Source**: {data_source}
            - **Forecast End Year**: {int(forecast_x[-1])}
            
            {"**Important Note**: This forecast assumes no major geopolitical shifts, policy changes, or economic crises. Actual spending may vary significantly based on global events, changes in threat perceptions, or budget constraints." if forecast_years > 3 else ""}
            
            {f"**Projected {forecast_years}-Year Change**: " + ('+' if forecast_y[-1] > primary_data['Spending (USD)'].iloc[-1] else '') + f"{(forecast_y[-1] - primary_data['Spending (USD)'].iloc[-1])/primary_data['Spending (USD)'].iloc[-1]*100:.1f}%" if data_source == "SIPRI" else f"**Projected {forecast_years}-Year Change**: " + ('+' if forecast_y[-1] > primary_data['Spending (% of GDP)'].iloc[-1] else '') + f"{forecast_y[-1] - primary_data['Spending (% of GDP)'].iloc[-1]:.2f} percentage points"}
            """)

with col2:
    spending_col = 'Spending (USD)' if data_source == 'SIPRI' else 'Spending (% of GDP)'
    country_data = df[df['Country'] == country].sort_values('Year')
    non_zero_data = country_data[country_data[spending_col] > 0]

    if not non_zero_data.empty:
        latest_data = non_zero_data.iloc[-1]
        latest_year = latest_data['Year']
        current_spending = latest_data[spending_col]

        if data_source == "SIPRI":
            st.markdown(f"""
            <div class="metric-card fade-in">
                <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;">Current Spending</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #00d2d3;">${current_spending/1000:,.1f}B</div>
                <div style="font-size: 0.9rem; color: #94a3b8;">{latest_year}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add forecast metric if enabled
            if show_forecast and 'show_forecast' in locals() and forecast_x is not None:
                forecast_end_year = int(forecast_x[-1])
                forecast_end_value = forecast_y[-1]
                
                forecast_change = (forecast_end_value - current_spending) / current_spending * 100
                change_color = "var(--positive)" if forecast_change >= 0 else "var(--negative)"
                
                st.markdown(f"""
                <div class="metric-card fade-in">
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;">Forecast ({forecast_end_year})</div>
                    <div style="font-size: 1.5rem; font-weight: 600; color: var(--forecast);">${forecast_end_value/1000:,.1f}B</div>
                    <div style="font-size: 0.9rem; color: {change_color};">{forecast_change:+.1f}% from {latest_year}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card fade-in">
                <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;">Current Spending</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #00d2d3;">{current_spending:.2f}% of GDP</div>
                <div style="font-size: 0.9rem; color: #94a3b8;">{latest_year}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add forecast metric for NATO data if enabled
            if show_forecast and 'show_forecast' in locals() and forecast_x is not None:
                forecast_end_year = int(forecast_x[-1])
                forecast_end_value = forecast_y[-1]
                
                # For NATO data, show whether the forecast meets the 2% target
                meets_target = forecast_end_value >= 2.0
                target_status = "Meets 2% target" if meets_target else "Below 2% target"
                target_color = "var(--positive)" if meets_target else "var(--negative)"
                
                # Calculate percentage points change
                forecast_change = forecast_end_value - current_spending
                change_color = "var(--positive)" if forecast_change >= 0 else "var(--negative)"
                
                st.markdown(f"""
                <div class="metric-card fade-in">
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;">Forecast ({forecast_end_year})</div>
                    <div style="font-size: 1.5rem; font-weight: 600; color: var(--forecast);">{forecast_end_value:.2f}% of GDP</div>
                    <div style="font-size: 0.9rem; color: {change_color};">{forecast_change:+.2f}pp from {latest_year}</div>
                    <div style="font-size: 0.9rem; color: {target_color}; margin-top: 4px;">{target_status}</div>
                </div>
                """, unsafe_allow_html=True)

        five_years_ago = latest_year - 5
        past_data = country_data[country_data['Year'] == five_years_ago]
        if not past_data.empty:
            spending_5y_ago = past_data.iloc[0][spending_col]
            if data_source == "NATO":
                change = current_spending - spending_5y_ago
                change_str = f"{change:+.2f}pp"
            else:
                change = (current_spending - spending_5y_ago) / spending_5y_ago * 100
                change_str = f"{change:+.1f}%"
            
            change_color = "var(--positive)" if change >= 0 else "var(--negative)"
            st.markdown(f"""
            <div class="metric-card fade-in">
                <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;">5-Year Change</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {change_color};">{change_str}</div>
                <div style="font-size: 0.9rem; color: #94a3b8;">{five_years_ago} → {latest_year}</div>
            </div>
            """, unsafe_allow_html=True)

    # Key Events Table - FIXED VERSION
    st.markdown("<h3 style='margin-top: 24px; margin-bottom: 12px;'>Key Events</h3>", unsafe_allow_html=True)
    
    event_impacts = []
    for year, event_name in EVENTS["Global"].items():
        impact = calculate_event_impact(country, year, df)
        if impact:
            event_impacts.append(impact)
    
    event_impacts.sort(key=lambda x: abs(x['change']), reverse=True)
    
    # Create DataFrame for table display
    if event_impacts:
        table_data = []
        for impact in event_impacts:
            change_class = "positive" if impact['change'] >= 0 else "negative"
            change_str = f"{impact['change']:+.2f}pp" if impact['is_nato'] else f"{impact['change']:+.1f}%"
            table_data.append({
                "Year": impact['year'],
                "Event": impact['name'],
                "Impact": change_str,
                "change_class": change_class
            })
        
        df_events = pd.DataFrame(table_data)
        
        # Apply custom styling to the table
        def color_impact(val):
            color = 'var(--positive)' if val == 'positive' else 'var(--negative)'
            return f'color: {color}'
        
        # Create styled dataframe
        styled_df = pd.DataFrame({
            'Year': df_events['Year'],
            'Event': df_events['Event'],
            'Impact': df_events['Impact']
        })
        
        # Apply styling using pandas Styler
        styler = styled_df.style.apply(
            lambda x: [
                'color: var(--positive)' if df_events.loc[i, 'change_class'] == 'positive' else 'color: var(--negative)' 
                for i in range(len(df_events))
            ], 
            subset=['Impact']
        ).set_properties(**{
            'text-align': 'right'
        }, subset=['Impact']).hide(axis="index")
        
        # Display table with proper styling
        st.write(styler.to_html(), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="width: 100%; text-align: center; padding: 16px; color: #94a3b8;">
            No event data available
        </div>
        """, unsafe_allow_html=True)

# Footer
st.divider()
st.caption("Data Sources: SIPRI Military Expenditure Database • NATO Annual Reports")

# Animations
html("""
<script>
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
        }
    });
}, {threshold: 0.1});

document.querySelectorAll('.fade-in').forEach(el => {
    el.style.opacity = 0;
    observer.observe(el);
});
</script>
""")