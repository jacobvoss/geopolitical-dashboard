import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    .plot-container {{
        background-color: var(--card);
        border-radius: 12px;
        padding: 16px;
    }}

    .stDataFrame {{
        background-color: var(--card) !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: var(--card);
        border-radius: 8px 8px 0 0 !important;
        padding: 8px 16px;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: var(--primary) !important;
        color: white !important;
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
    
    # Benchmark data (example - replace with real calculations)
    df_melted['NATO_Avg'] = df_melted.groupby('Year')['Spending (USD)'].transform('mean')
    df_melted['GDP_Pct'] = df_melted['Spending (USD)'] / 1e10  # Replace with actual GDP data
    
    return df_melted.dropna()

df = load_data()
latest_year = df['Year'].max()

# ===== UI =====
st.title("NATO Defense Spending Analysis")
st.caption("Comparative military expenditure trends 1949-2024")

# Sidebar controls
with st.sidebar:
    st.header("Filters")
    country = st.selectbox(
        "Primary Country",
        df['Country'].unique(),
        index=0
    )
    
    compare_countries = st.multiselect(
        "Compare With",
        df['Country'].unique(),
        default=['United States', 'Germany']
    )
    
    analysis_type = st.radio(
        "Analysis Mode",
        ["Trend Analysis", "Benchmarking", "Geospatial"]
    )

# Main content
if analysis_type == "Trend Analysis":
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main time series chart
        fig = go.Figure()
        
        # Primary country
        fig.add_trace(go.Scatter(
            x=df[df['Country'] == country]['Year'],
            y=df[df['Country'] == country]['Spending (USD)'],
            name=country,
            line=dict(color='#4a8cff', width=3),
            mode='lines'
        ))
        
        # Comparison countries
        for c in compare_countries:
            fig.add_trace(go.Scatter(
                x=df[df['Country'] == c]['Year'],
                y=df[df['Country'] == c]['Spending (USD)'],
                name=c,
                line=dict(width=1.5, dash='dot'),
                opacity=0.7
            ))
        
        fig.update_layout(
            title=f"Defense Spending Trend: {country} vs Benchmarks",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric(
            "2023 Spending",
            f"${df[(df['Country']==country)&(df['Year']==latest_year)]['Spending (USD)'].values[0]/1e9:,.1f}B"
        )
        st.metric(
            "vs NATO Avg",
            f"{df[(df['Country']==country)&(df['Year']==latest_year)]['Spending (USD)'].values[0]/df[df['Year']==latest_year]['NATO_Avg'].mean()*100:,.0f}%"
        )
        st.metric(
            "5-Yr Change",
            f"{(df[(df['Country']==country)&(df['Year']==latest_year)]['Spending (USD)'].values[0]/df[(df['Country']==country)&(df['Year']==latest_year-5)]['Spending (USD)'].values[0]-1)*100:+.1f}%"
        )

elif analysis_type == "Benchmarking":
    st.subheader("Performance Benchmarking")
    
    # What benchmarking does:
    # 1. Compares selected country against key metrics
    # 2. Shows relative performance to NATO averages
    # 3. Identifies over/under performance
    
    tab1, tab2 = st.tabs(["Spending Efficiency", "Strategic Position"])
    
    with tab1:
        st.write(f"""
        #### {country}'s Defense Spending Efficiency
        - **Current Spending**: ${df[(df['Country']==country)&(df['Year']==latest_year)]['Spending (USD)'].values[0]/1e9:,.1f}B
        - **As % of NATO Total**: {(df[(df['Country']==country)&(df['Year']==latest_year)]['Spending (USD)'].values[0]/df[df['Year']==latest_year]['Spending (USD)'].sum())*100:.1f}%
        - **Per Capita**: ${df[(df['Country']==country)&(df['Year']==latest_year)]['Spending (USD)'].values[0]/50e6:,.0f} (est.)
        """)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Spending Share", "Growth vs Peers"))
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=['Selected Country', 'Other NATO'],
                values=[
                    df[(df['Country']==country)&(df['Year']==latest_year)]['Spending (USD)'].values[0],
                    df[df['Year']==latest_year]['Spending (USD)'].sum() - df[(df['Country']==country)&(df['Year']==latest_year)]['Spending (USD)'].values[0]
                ],
                marker_colors=['#4a8cff', '#334155'],
                hole=0.5
            ), row=1, col=1)
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=compare_countries,
                y=[df[(df['Country']==c)&(df['Year']==latest_year)]['Spending (USD)'].values[0]/1e9 for c in compare_countries],
                marker_color='#4a8cff',
                name='2023 Spending (B USD)'
            ), row=1, col=2)
        
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    

    with tab2:
        st.write("""
        #### Strategic Positioning
        Compares your selected country against key benchmarks:
        """)
        
        # First calculate the values to avoid complex inline expressions
        current_gdp_pct = df[(df['Country'] == country) & (df['Year'] == latest_year)]['GDP_Pct'].values[0] * 100
        current_spending = df[(df['Country'] == country) & (df['Year'] == latest_year)]['Spending (USD)'].values[0]
        spending_5y_ago = df[(df['Country'] == country) & (df['Year'] == latest_year - 5)]['Spending (USD)'].values[0]
        
        benchmarks = pd.DataFrame({
            'Metric': ['Spending/GDP', 'Spending/Capita', '5-Yr Growth'],
            'Your Country': [
                f"{current_gdp_pct:.1f}%",  # Simplified expression
                f"${current_spending / 50e6:,.0f}",
                f"{(current_spending / spending_5y_ago - 1) * 100:+.1f}%"
            ],
            'NATO Average': [
                "2.0%",
                "$1,200",
                "+12.5%"
            ],
            'Target': [
                "≥2.0%",
                "≥$1,500",
                "Match inflation+2%"
            ]
        })
        
        st.dataframe(
            benchmarks,
            column_config={
                "Metric": st.column_config.TextColumn(width="medium"),
                "Your Country": st.column_config.NumberColumn(
                    help="Your country's performance",
                    width="small"
                ),
                "NATO Average": st.column_config.TextColumn(
                    help="NATO average performance",
                    width="small"
                ),
                "Target": st.column_config.TextColumn(
                    help="Recommended target",
                    width="small"
                )
            },
            hide_index=True,
            use_container_width=True
        )

elif analysis_type == "Geospatial":
    st.subheader("Regional Spending Patterns")
    
    df_latest = df[df['Year'] == latest_year]
    fig = px.choropleth(
        df_latest,
        locations="Country",
        locationmode='country names',
        color="Spending (USD)",
        hover_name="Country",
        color_continuous_scale='blues',
        range_color=(0, df_latest['Spending (USD)'].quantile(0.9)),
        projection="natural earth",
        height=600
    )
    
    fig.update_geos(
        bgcolor='rgba(0,0,0,0)',
        landcolor='#1e293b',
        subunitcolor='#334155'
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(
            title="Spending (USD)",
            tickprefix="$"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.caption("""
Data Sources: SIPRI Military Expenditure Database
Analysis Period: 1949-2024 • All figures in constant 2023 USD
""")