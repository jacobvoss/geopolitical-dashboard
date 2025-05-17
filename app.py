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

# ===== UI =====
st.title("NATO Defense Spending Analysis")
st.caption("Comparative military expenditure trends 1949-2024")

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
    current_spending = df[(df['Country'] == country) & (df['Year'] == latest_year)]['Spending (USD)'].values[0]
    st.metric(
        f"{latest_year} Spending", 
        f"${current_spending/1e9:,.1f}B"
    )
    
    # Calculate 10-year change if data exists
    if latest_year - 10 in df['Year'].unique():
        spending_10y_ago = df[(df['Country'] == country) & (df['Year'] == latest_year - 10)]['Spending (USD)'].values[0]
        pct_change = (current_spending - spending_10y_ago) / spending_10y_ago * 100
        st.metric(
            "10-Year Change",
            f"{pct_change:+.1f}%",
            delta_color="inverse"
        )

# Footer
st.divider()
st.caption("Data Sources: SIPRI Military Expenditure Database • 2024")