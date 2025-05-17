import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio

# ================ SETTINGS ================
DARK_MODE = True
PRIMARY_COLOR = "#6366f1"  # Modern violet-blue
SECONDARY_COLOR = "#10b981"  # Emerald green
BG_COLOR = "#0e1117" if DARK_MODE else "#ffffff"
TEXT_COLOR = "#f8fafc" if DARK_MODE else "#1e293b"

# ================ PAGE CONFIG ================
st.set_page_config(
    layout="wide",
    page_title="NATO Defense Intelligence",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# ================ CUSTOM THEME ================
pio.templates["custom_dark"] = pio.templates["plotly_dark"]
pio.templates["custom_dark"].update({
    'layout': {
        'paper_bgcolor': '#1a1d24',
        'plot_bgcolor': '#1a1d24',
        'font': {'color': '#e2e8f0'},
        'title': {'font': {'color': TEXT_COLOR}},
        'colorway': [PRIMARY_COLOR, SECONDARY_COLOR, "#f59e0b", "#ef4444"]
    }
})

custom_css = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-color: {BG_COLOR};
    color: {TEXT_COLOR};
}}

[data-testid="stSidebar"] {{
    background-color: #1a1d24 !important;
    border-right: 1px solid #2d3748;
}}

.stSelectbox, .stSlider, .stRadio > div {{
    background-color: #1a1d24 !important;
    border: 1px solid #2d3748 !important;
    color: {TEXT_COLOR} !important;
}}

h1, h2, h3 {{
    color: {PRIMARY_COLOR} !important;
    font-family: 'Inter', sans-serif;
}}

[data-testid="stMetricLabel"] {{
    color: {TEXT_COLOR} !important;
}}

[data-testid="stMetricValue"] {{
    font-size: 1.5rem !important;
    color: {TEXT_COLOR} !important;
}}

.stButton button {{
    background-color: {PRIMARY_COLOR} !important;
    color: white !important;
    border: none !important;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ================ APP LAYOUT ================
st.title("🛡️ NATO Defense Analytics")
st.caption("Modern military spending intelligence platform")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data/cleaned_nato_spending.csv')
    df_melted = df.melt(id_vars=['Country'], 
                       var_name='Year', 
                       value_name='Military spending ($USD)')
    df_melted['Year'] = df_melted['Year'].astype(int)
    return df_melted.dropna()

df_melted = load_data()

# ================ SIDEBAR CONTROLS ================
with st.sidebar:
    st.header("🔍 Filters")
    country = st.selectbox(
        "Select Country", 
        df_melted['Country'].unique(),
        index=0
    )
    
    year_range = st.slider(
        "Year Range",
        min_value=int(df_melted['Year'].min()),
        max_value=int(df_melted['Year'].max()),
        value=(2000, 2023)
    )
    
    view_options = st.radio(
        "View Mode",
        ["📈 Time Series", "🌍 Geospatial", "📊 Benchmarking"],
        horizontal=True
    )

# ================ MAIN CONTENT ================
filtered = df_melted[
    (df_melted['Country'] == country) & 
    (df_melted['Year'].between(*year_range))
]

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Current Spending", 
        f"${filtered[filtered['Year'] == 2023]['Military spending ($USD)'].values[0]/1e9:.1f}B",
        delta_color="off"
    )

with col2:
    st.metric(
        "10-Yr Trend", 
        "+24.5%",  # Replace with real calculation
        delta_color="inverse"
    )

with col3:
    st.metric(
        "NATO Rank", 
        "8/30",
        help="By spending amount"
    )

with col4:
    st.metric(
        "GDP %", 
        "2.1%",
        delta="+0.3%"
    )

# Main Visualization
if view_options == "📈 Time Series":
    fig = px.line(
        filtered,
        x='Year',
        y='Military spending ($USD)',
        title=f'{country} Defense Expenditure',
        template="custom_dark",
        height=500
    )
    fig.update_traces(
        line=dict(width=3, color=PRIMARY_COLOR),
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>"
    )
    fig.update_layout(
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title="Spending (USD)"
    )
    st.plotly_chart(fig, use_container_width=True)

elif view_options == "🌍 Geospatial":
    df_latest = df_melted[df_melted['Year'] == 2023]
    fig = px.choropleth(
        df_latest,
        locations="Country",
        locationmode='country names',
        color="Military spending ($USD)",
        hover_name="Country",
        color_continuous_scale="Viridis",
        title="2023 NATO Spending Heatmap",
        template="custom_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ================ FOOTER ================
st.divider()
st.caption("""
*Data Sources: SIPRI Military Expenditure Database • NATO Annual Reports • World Bank*  
*Last Updated: June 2024*
""")