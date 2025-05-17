import streamlit as st
import pandas as pd
import plotly.express as px

# MUST be first Streamlit command
st.set_page_config(
    layout="wide",
    page_title="Geopolitical Risk Dashboard",
    page_icon=":military_helmet:",  # or "🪖" or "assets/helmet.png"
    initial_sidebar_state="expanded"
)

# Custom CSS for military theme
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f0f2f6;
    background-image: url("https://www.transparenttextures.com/patterns/concrete-wall.png");
}
.stSelectbox, .stMultiSelect {
    background-color: #ffffff !important;
    border: 1px solid #d6d6d6;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
[data-testid="stToolbar"] {
    right: 2rem;
}
h1 {
    color: #2a3439;
    border-bottom: 2px solid #8b0000;
}
</style>
""", unsafe_allow_html=True)

# Main App
st.title("🌍 Geopolitical Risk Dashboard")
st.caption("Tracking defense spending patterns across NATO members")

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data/cleaned_nato_spending.csv')
    df_melted = df.melt(id_vars=['Country'],
                       var_name='Year',
                       value_name='Military spending ($USD)')
    df_melted['Year'] = df_melted['Year'].astype(int)
    return df_melted.dropna()

df_melted = load_data()

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    country = st.selectbox("Choose a country", df_melted['Country'].unique())
    show_map = st.checkbox("Show world map comparison", True)
    metric = st.radio("View as:", ["Absolute ($)", "% of GDP"])

# Main content tabs
tab1, tab2 = st.tabs(["📈 Time Series", "🗺️ Geospatial View"])

# Time Series Tab
with tab1:
    filtered = df_melted[df_melted['Country'] == country]
    
    fig = px.line(
        filtered,
        x='Year',
        y='Military spending ($USD)',
        title=f'{country} Defense Spending Over Time',
        color_discrete_sequence=["#8b0000"],  # Military red
        template="plotly_white"
    )
    fig.update_layout(
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

# Map Tab
with tab2:
    if show_map:
        df_latest = df_melted[df_melted['Year'] == df_melted['Year'].max()]
        fig = px.choropleth(
            df_latest,
            locations="Country",
            locationmode='country names',
            color="Military spending ($USD)",
            hover_name="Country",
            color_continuous_scale="OrRd",
            title="Latest NATO Military Spending by Country"
        )
        st.plotly_chart(fig, use_container_width=True)

# Key Metrics
col1, col2, col3 = st.columns(3)
latest_year = df_melted['Year'].max()
country_data = df_melted[(df_melted['Country'] == country) & 
                        (df_melted['Year'] == latest_year)]

with col1:
    st.metric(
        f"{latest_year} Spending", 
        f"${country_data['Military spending ($USD)'].values[0]/1e9:.2f}B",
        help="In current USD"
    )

with col2:
    st.metric(
        "10-Yr Change",
        f"+15.3%",  # Replace with real calculation
        delta_color="inverse"
    )

with col3:
    st.metric(
        "NATO Rank",
        "8/30",
        help="By spending amount"
    )