import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
        --primary: #4a8cff;
        --secondary: #ff6b4a;
        --bg: #0f172a;
        --card: #1e293b;
        --text: #e2e8f0;
    }}

    [data-testid="stAppViewContainer"] > .main {{
        background-color: var(--bg);
    }}

    [data-testid="stSidebar"] {{
        background-color: var(--card) !important;
        border-right: 1px solid rgba(74, 140, 255, 0.1);
    }}

    .stSelectbox, .stSlider, .stRadio > div {{
        background-color: var(--card) !important;
        border: 1px solid #334155 !important;
    }}

    h1, h2, h3 {{
        color: var(--text);
        font-weight: 500;
    }}

    [data-testid="stMetricValue"] {{
        font-size: 1.5rem;
        color: var(--primary) !important;
    }}
    
    .event-annotation {{
        background-color: rgba(255,107,74,0.2);
        padding: 4px 8px;
        border-radius: 4px;
        border-left: 3px solid var(--secondary);
        margin: 8px 0;
    }}
    </style>
    """, unsafe_allow_html=True)

apply_styles()

# ===== DATA =====
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data/cleaned_nato_spending.csv')
    df_melted = df.melt(id_vars=['Country'], 
                       var_name='Year', 
                       value_name='Spending (USD)')
    df_melted['Year'] = df_melted['Year'].astype(int)
    return df_melted.dropna()

df = load_data()
latest_year = df['Year'].max()
available_countries = df['Country'].unique().tolist()

# Historical events data
EVENTS = {
    "Global": {
        2001: "9/11 Attacks",
        2008: "Global Financial Crisis",
        2014: "Crimea Annexation",
        2020: "COVID-19 Pandemic",
        2022: "Russia Invades Ukraine"
    },
    "United States": {
        2003: "Iraq War",
        2011: "Bin Laden Killed",
        2017: "Trump Defense Boost"
    },
    "Germany": {
        2011: "Military Reform",
        2016: "Defense Spending Increase"
    }
}

# ===== UI =====
st.title("NATO Defense Spending Analysis")
st.caption("Comparative military expenditure trends 1949-2024 with event annotations")

# Sidebar controls
with st.sidebar:
    st.header("Filters")
    country = st.selectbox(
        "Primary Country",
        available_countries,
        index=available_countries.index('United States') if 'United States' in available_countries else 0
    )
    
    compare_countries = st.multiselect(
        "Compare With",
        available_countries,
        default=['Germany', 'France'] if set(['Germany', 'France']).issubset(available_countries) else []
    )
    
    show_events = st.checkbox("Show historical events", True)

# Main visualization
col1, col2 = st.columns([3, 1])

with col1:
    fig = go.Figure()
    
    # Primary country
    primary_data = df[df['Country'] == country]
    fig.add_trace(go.Scatter(
        x=primary_data['Year'],
        y=primary_data['Spending (USD)'],
        name=country,
        line=dict(color='#4a8cff', width=3),
        mode='lines'
    ))
    
    # Comparison countries
    for c in compare_countries:
        comp_data = df[df['Country'] == c]
        fig.add_trace(go.Scatter(
            x=comp_data['Year'],
            y=comp_data['Spending (USD)'],
            name=c,
            line=dict(width=1.5, dash='dot'),
            opacity=0.7
        ))
    
    # Add event annotations if enabled
    if show_events:
        # Global events
        for year, event in EVENTS.get("Global", {}).items():
            if year in primary_data['Year'].values:
                fig.add_vline(
                    x=year,
                    line_width=1,
                    line_dash="dash",
                    line_color="#ff6b4a",
                    opacity=0.5,
                    annotation_text=event,
                    annotation_position="top left"
                )
        
        # Country-specific events
        for year, event in EVENTS.get(country, {}).items():
            if year in primary_data['Year'].values:
                fig.add_vline(
                    x=year,
                    line_width=1,
                    line_dash="dash",
                    line_color="#10b981",
                    opacity=0.5,
                    annotation_text=event,
                    annotation_position="bottom right"
                )
    
    fig.update_layout(
        title=f"{country} Defense Spending Over Time",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Get latest non-zero spending value
    country_data = df[df['Country'] == country]
    latest_non_zero = country_data[country_data['Spending (USD)'] > 0].iloc[-1]
    
    st.metric(
        f"Latest Data ({latest_non_zero['Year']})", 
        f"${latest_non_zero['Spending (USD)']/1e9:,.1f}B"
    )
    
    # Calculate 10-year change if data exists
    if len(country_data) >= 10:
        current_year = latest_non_zero['Year']
        current_spending = latest_non_zero['Spending (USD)']
        ten_years_ago = max(current_year - 10, country_data['Year'].min())
        
        try:
            spending_10y_ago = country_data[country_data['Year'] == ten_years_ago]['Spending (USD)'].values[0]
            pct_change = (current_spending - spending_10y_ago) / spending_10y_ago * 100
            st.metric(
                "10-Year Change",
                f"{pct_change:+.1f}%",
                delta_color="inverse"
            )
        except:
            pass
    
    # Event explanations
    if show_events:
        st.markdown("### Key Events")
        country_events = {**EVENTS.get("Global", {}), **EVENTS.get(country, {})}
        
        for year, event in sorted(country_events.items()):
            if year in country_data['Year'].values:
                spending = country_data[country_data['Year'] == year]['Spending (USD)'].values[0]
                st.markdown(
                    f"""<div class="event-annotation">
                    <strong>{year}: {event}</strong><br>
                    Spending: ${spending/1e9:,.1f}B
                    </div>""",
                    unsafe_allow_html=True
                )

# Footer
st.divider()
st.caption("Data Sources: SIPRI Military Expenditure Database • 2024")