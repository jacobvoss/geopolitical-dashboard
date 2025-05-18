import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit.components.v1 import html

# ===== CONFIG =====
st.set_page_config(
    layout="wide",
    page_title="NATO Defense Analytics",
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
    ...
    </style>
    """, unsafe_allow_html=True)

    # Add Inter font
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

apply_styles()

# ===== DATA LOADING =====
@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename)

    # Melt the dataframe
    df_melted = df.melt(id_vars=['Country'], 
                        var_name='Year', 
                        value_name='Spending (USD)')

    # Filter out rows where Year isn't a digit
    df_melted = df_melted[df_melted['Year'].str.match(r'^\d{4}$', na=False)]

    df_melted['Year'] = df_melted['Year'].astype(int)
    df_melted['Spending (USD)'] = pd.to_numeric(df_melted['Spending (USD)'], errors='coerce')

    # Sort and calculate year-on-year change
    df_melted = df_melted.sort_values(['Country', 'Year'])
    df_melted['YoY_Change'] = df_melted.groupby('Country')['Spending (USD)'].pct_change() * 100
    return df_melted.dropna()


# ===== DATASET SELECTION =====
with st.sidebar:
    st.header("Filters")
    data_version = st.radio(
        "Dataset Version",
        ["Original", "Updated"],
        horizontal=True
    )

# File paths
if data_version == "Original":
    df = load_data("cleaned_data/SIPRI_spending_clean.csv")
    version_label = "Original Dataset"
else:
    df = load_data("cleaned_data/nato_defense_spending_clean.csv")
    version_label = "Updated Dataset"

available_countries = df['Country'].unique().tolist()
latest_year = df['Year'].max()

# ===== EVENT IMPACT SETUP =====
EVENTS = {
    "Global": {
        2001: "9/11 Attacks",
        2008: "Global Financial Crisis",
        2014: "Crimea Annexation",
        2020: "COVID-19 Pandemic",
        2022: "Russia Invades Ukraine"
    }
}

def calculate_event_impact(country, year, df):
    country_data = df[df['Country'] == country].sort_values('Year').reset_index(drop=True)
    try:
        event_idx = country_data.index[country_data['Year'] == year][0]
    except IndexError:
        return None

    if event_idx == 0:
        return None

    yoy_change = country_data.at[event_idx, 'YoY_Change']
    if pd.isna(yoy_change) or abs(yoy_change) == float('inf'):
        return None

    return {
        'year': year,
        'name': EVENTS.get("Global", {}).get(year),
        'change': yoy_change,
        'prev_year': country_data.at[event_idx - 1, 'Year']
    }

# ===== UI: COUNTRY SELECTORS =====
country = st.sidebar.selectbox(
    "Select Country",
    available_countries,
    index=available_countries.index('United States') if 'United States' in available_countries else 0
)

compare_countries = st.sidebar.multiselect(
    "Compare With",
    [c for c in available_countries if c != country]
)

# ===== PAGE TITLE =====
st.title("Defense Spending Analytics")
st.caption(f"{version_label} • NATO military expenditure trends with event impact analysis")

# ===== MAIN VISUALIZATION =====
col1, col2 = st.columns([3, 1])

with col1:
    fig = go.Figure()

    # Main line
    primary_data = df[df['Country'] == country]
    fig.add_trace(go.Scatter(
        x=primary_data['Year'],
        y=primary_data['Spending (USD)'],
        name=country,
        line=dict(color='#00d2d3', width=3),
        mode='lines',
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>"
    ))

    # Comparison lines
    if compare_countries:
        line_styles = ['solid', 'dash', 'dot', 'dashdot']
        colors = ['#ff6b4a', '#f0a202', '#a1cdf4', '#d883ff']
        for i, c in enumerate(compare_countries):
            comp_data = df[df['Country'] == c]
            fig.add_trace(go.Scatter(
                x=comp_data['Year'],
                y=comp_data['Spending (USD)'],
                name=c,
                line=dict(
                    width=2.5,
                    dash=line_styles[i % len(line_styles)],
                    color=colors[i % len(colors)]
                ),
                opacity=0.9,
                hovertemplate="<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>"
            ))

    # Event annotations
    for year, event_name in sorted(EVENTS["Global"].items()):
        impact = calculate_event_impact(country, year, df)
        if impact and impact['change'] is not None:
            fig.add_vline(
                x=year,
                line_width=1,
                line_dash="dash",
                line_color="#ff6b4a",
                opacity=0.5,
                annotation_text=f"{event_name}<br>{'+' if impact['change'] >= 0 else ''}{impact['change']:.1f}%",
                annotation_position="top right",
                annotation_font_size=10,
                annotation_bgcolor="rgba(30,41,59,0.8)"
            )

    fig.update_layout(
        title=f"{country} Defense Spending ({version_label})",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family="Inter"),
        hovermode="x unified",
        height=500,
        margin=dict(t=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    country_data = df[df['Country'] == country]
    country_data['Spending (USD)'] = pd.to_numeric(country_data['Spending (USD)'], errors='coerce')
    non_zero_data = country_data[country_data['Spending (USD)'] > 0]

    if not non_zero_data.empty:
        latest_data = non_zero_data.sort_values('Year').iloc[-1]
        latest_year = latest_data['Year']
        current_spending = latest_data['Spending (USD)']

        st.markdown(f"""
        <div class="metric-card fade-in">
            <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;">Current Spending</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #00d2d3;">${current_spending/1e9:,.1f}B</div>
            <div style="font-size: 0.9rem; color: #94a3b8;">{latest_year}</div>
        </div>
        """, unsafe_allow_html=True)

        if len(country_data) >= 5:
            five_years_ago = max(latest_year - 5, country_data['Year'].min())
            past_data = country_data[country_data['Year'] == five_years_ago]

            if not past_data.empty:
                spending_5y_ago = past_data.iloc[0]['Spending (USD)']
                pct_change = (current_spending - spending_5y_ago) / spending_5y_ago * 100
                change_color = "var(--positive)" if pct_change >= 0 else "var(--negative)"

                st.markdown(f"""
                <div class="metric-card fade-in">
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;">5-Year Change</div>
                    <div style="font-size: 1.5rem; font-weight: 600; color: {change_color};">
                        {'+' if pct_change >= 0 else ''}{pct_change:.1f}%
                    </div>
                    <div style="font-size: 0.9rem; color: #94a3b8;">{five_years_ago} → {latest_year}</div>
                </div>
                """, unsafe_allow_html=True)

    # Timeline of events
    st.markdown("""
    <div class="fade-in">
        <h3 style="margin-top: 24px; margin-bottom: 12px;">Key Events</h3>
        <table class="event-table">
            <thead>
                <tr>
                    <th>Year</th>
                    <th>Event</th>
                    <th style="text-align: right;">Impact</th>
                </tr>
            </thead>
            <tbody>
    """, unsafe_allow_html=True)

    event_impacts = []
    for year, event_name in EVENTS["Global"].items():
        impact = calculate_event_impact(country, year, df)
        if impact and impact['change'] is not None:
            event_impacts.append(impact)

    event_impacts.sort(key=lambda x: abs(x['change']), reverse=True)

    for impact in event_impacts:
        change_class = "positive-change" if impact['change'] >= 0 else "negative-change"
        change_text = f"{'+' if impact['change'] >= 0 else ''}{impact['change']:.1f}%"
        st.markdown(f"""
        <tr class="fade-in">
            <td>{impact['year']}</td>
            <td>{impact['name']}</td>
            <td style="text-align: right;" class="{change_class}">{change_text}</td>
        </tr>
        """, unsafe_allow_html=True)

    if not event_impacts:
        st.markdown("""
        <tr class="fade-in">
            <td colspan="3" style="text-align: center; color: #94a3b8;">No event data available</td>
        </tr>
        """, unsafe_allow_html=True)

    st.markdown("""
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.divider()
st.caption("Data Sources: SIPRI Military Expenditure Database • NATO Annual Reports")

# Animation Script
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
