"""Interactive dashboard for exploring and forecasting NATO and SIPRI defence
spending data.

The app loads preprocessed datasets, builds forecasts with several models and
visualizes spending trends alongside notable geopolitical events.
"""

import streamlit as st
import pandas as pd
from streamlit.components.v1 import html

from src.data import load_data
from src.visualization import create_spending_chart

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
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)


apply_styles()




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

    st.header("Forecast")
    forecast_years = st.slider("Years to Forecast", min_value=1, max_value=10, value=5)
    forecast_model = st.selectbox("Forecast Model", ['ARIMA', 'ExponentialSmoothing', 'LinearRegression'])

# Main layout
col1, col2 = st.columns([3, 1])

with col1:
    fig = create_spending_chart(
        df,
        data_source,
        country,
        compare_countries,
        forecast_years,
        forecast_model,
    )
    st.plotly_chart(fig, use_container_width=True)

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
                <div style="font-size: 1.5rem; font-weight: 600; color: #00d2d3;">${current_spending / 1000:,.1f}B</div>
                <div style="font-size: 0.9rem; color: #94a3b8;">{latest_year}</div>
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
    st.markdown("""
    <h3 style='margin-top: 24px; margin-bottom: 12px; display: flex; align-items: center; gap: 6px;'>
        Key Events
        <span title="Impacts are approximate and may not be directly caused by these events." style="cursor: help; font-size: 0.85em; color: #94a3b8;">ⓘ</span>
    </h3>
    """, unsafe_allow_html=True)

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
