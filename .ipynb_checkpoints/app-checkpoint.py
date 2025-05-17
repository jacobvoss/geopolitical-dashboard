import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit.components.v1 import html
from datetime import datetime

# ===== CONFIG =====
st.set_page_config(
    layout="wide",
    page_title="NATO Defense Analytics",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# ===== STYLES & ANIMATIONS =====
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

    [data-testid="stAppViewContainer"] > .main {{
        background-color: var(--bg);
        font-family: 'Inter', sans-serif;
    }}

    [data-testid="stSidebar"] {{
        background-color: var(--card) !important;
        border-right: 1px solid rgba(0, 210, 211, 0.1);
    }}

    .stSelectbox, .stSlider, .stRadio > div {{
        background-color: var(--card) !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }}

    h1, h2, h3 {{
        color: var(--text);
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }}

    [data-testid="stMetricValue"] {{
        font-size: 1.5rem;
        color: var(--primary) !important;
        font-weight: 600;
    }}

    .event-table {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'Inter', sans-serif;
    }}
    
    .event-table tr {{
        transition: all 0.2s ease;
    }}
    
    .event-table tr:hover {{
        background-color: rgba(0, 210, 211, 0.05);
    }}
    
    .event-table td {{
        padding: 10px 12px;
        border-bottom: 1px solid #334155;
        font-size: 0.9rem;
    }}
    
    .event-table th {{
        padding: 8px 12px;
        text-align: left;
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 1px solid #334155;
    }}
    
    .positive-change {{
        color: var(--positive) !important;
        font-weight: 500;
    }}
    
    .negative-change {{
        color: var(--negative) !important;
        font-weight: 500;
    }}
    
    .metric-card {{
        background-color: var(--card);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        border-left: 4px solid var(--primary);
    }}
    
    .projection-card {{
        background-color: var(--card);
        border-radius: 12px;
        padding: 16px;
        margin-top: 24px;
        border: 1px solid rgba(0, 210, 211, 0.2);
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(5px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.3s ease-out forwards;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Add Inter font
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
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
    
    # Calculate year-over-year changes
    df_melted = df_melted.sort_values(['Country', 'Year'])
    df_melted['YoY_Change'] = df_melted.groupby('Country')['Spending (USD)'].pct_change() * 100
    
    # Generate 5-year projections (simplified example)
    current_year = datetime.now().year
    for y in range(current_year, current_year + 5):
        projection = df_melted[df_melted['Year'] == current_year - 1].copy()
        projection['Year'] = y
        projection['Spending (USD)'] = projection['Spending (USD)'] * 1.03  # 3% growth projection
        projection['YoY_Change'] = 3.0
        projection['is_projection'] = True
        df_melted = pd.concat([df_melted, projection])
    
    return df_melted.dropna()

df = load_data()
latest_year = df[~df['is_projection']]['Year'].max() if 'is_projection' in df.columns else df['Year'].max()
available_countries = df['Country'].unique().tolist()

# Historical events data with impact
EVENTS = {
    "Global": {
        2008: {"name": "Global Financial Crisis", "impact": -6.2},
        2014: {"name": "Crimea Annexation", "impact": 7.3},
        2020: {"name": "COVID-19 Pandemic", "impact": 4.7},
        2022: {"name": "Russia Invades Ukraine", "impact": 12.1}
    },
    "United States": {
        2001: {"name": "9/11 Attacks", "impact": 15.3},
        2003: {"name": "Iraq War", "impact": 18.2}
    }
}

# ===== UI =====
st.title("Defense Spending Analytics")
st.caption("NATO military expenditure trends and projections")

# Sidebar controls
with st.sidebar:
    st.header("Filters")
    country = st.selectbox(
        "Select Country",
        available_countries,
        index=available_countries.index('United States') if 'United States' in available_countries else 0
    )
    
    compare_countries = st.multiselect(
        "Compare With",
        [c for c in available_countries if c != country],
        default=['Germany', 'France'] if set(['Germany', 'France']).issubset(available_countries) else []
    )
    
    st.markdown("---")
    st.markdown("### Key Metrics", class_="fade-in")

# Main visualization
col1, col2 = st.columns([3, 1])

with col1:
    fig = go.Figure()
    
    # Primary country - actual data
    primary_data = df[(df['Country'] == country) & (~df.get('is_projection', False))]
    fig.add_trace(go.Scatter(
        x=primary_data['Year'],
        y=primary_data['Spending (USD)'],
        name=country,
        line=dict(color='#00d2d3', width=3),
        mode='lines',
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>"
    ))
    
    # Primary country - projections
    if 'is_projection' in df.columns:
        projection_data = df[(df['Country'] == country) & (df['is_projection'])]
        fig.add_trace(go.Scatter(
            x=projection_data['Year'],
            y=projection_data['Spending (USD)'],
            name=f"{country} (Projection)",
            line=dict(color='#00d2d3', width=3, dash='dot'),
            mode='lines',
            hovertemplate="<b>%{x}</b><br>$%{y:,.0f}M (Projected)<extra></extra>"
        ))
    
    # Comparison countries
    for c in compare_countries:
        comp_data = df[(df['Country'] == c) & (~df.get('is_projection', False))]
        fig.add_trace(go.Scatter(
            x=comp_data['Year'],
            y=comp_data['Spending (USD)'],
            name=c,
            line=dict(width=1.5, dash='dot', color='#94a3b8'),
            opacity=0.8,
            hovertemplate="<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>"
        ))
    
    # Add event annotations
    all_events = {**EVENTS.get("Global", {}), **EVENTS.get(country, {})}
    for year, event in sorted(all_events.items()):
        if year in primary_data['Year'].values:
            fig.add_vline(
                x=year,
                line_width=1,
                line_dash="dash",
                line_color="#ff6b4a",
                opacity=0.5,
                annotation_text=f"{event['name']}<br>{'+' if event['impact'] >= 0 else ''}{event['impact']}%",
                annotation_position="top right",
                annotation_font_size=10,
                annotation_bgcolor="rgba(30,41,59,0.8)"
            )
    
    fig.update_layout(
        title=f"{country} Defense Spending",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family="Inter"),
        hovermode="x unified",
        height=500,
        margin=dict(t=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Current spending metric
    current_data = df[(df['Country'] == country) & (~df.get('is_projection', False)) & (df['Year'] == latest_year)]
    if not current_data.empty:
        current_spending = current_data['Spending (USD)'].values[0]
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;">Current Spending</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #00d2d3;">${current_spending/1e9:,.1f}B</div>
            <div style="font-size: 0.9rem; color: #94a3b8;">{latest_year}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 5-year change metric
    if len(df[df['Country'] == country]) >= 5:
        five_years_ago = latest_year - 5
        spending_5y_ago = df[(df['Country'] == country) & (df['Year'] == five_years_ago)]['Spending (USD)'].values[0]
        pct_change = (current_spending - spending_5y_ago) / spending_5y_ago * 100
        change_color = "var(--positive)" if pct_change >= 0 else "var(--negative)"
        
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;">5-Year Change</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: {change_color};">{'+' if pct_change >= 0 else ''}{pct_change:.1f}%</div>
            <div style="font-size: 0.9rem; color: #94a3b8;">{five_years_ago} → {latest_year}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Event timeline
    st.markdown("""
    <div class="fade-in">
        <h3 style="margin-top: 24px; margin-bottom: 12px;">Key Events</h3>
        <table class="event-table">
            <thead>
                <tr>
                    <th>Year</th>
                    <th>Event</th>
                    <th>Impact</th>
                </tr>
            </thead>
            <tbody>
    """, unsafe_allow_html=True)
    
    # Display events in table
    all_events = {**EVENTS.get("Global", {}), **EVENTS.get(country, {})}
    for year, event in sorted(all_events.items(), reverse=True):
        if year in df[df['Country'] == country]['Year'].values:
            change_class = "positive-change" if event['impact'] >= 0 else "negative-change"
            change_text = f"+{event['impact']}%" if event['impact'] >= 0 else f"{event['impact']}%"
            
            st.markdown(f"""
            <tr class="fade-in">
                <td>{year}</td>
                <td>{event['name']}</td>
                <td class="{change_class}">{change_text}</td>
            </tr>
            """, unsafe_allow_html=True)
    
    st.markdown("""
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # 5-year projections
    if 'is_projection' in df.columns:
        st.markdown("""
        <div class="projection-card fade-in">
            <h3 style="margin-top: 0; margin-bottom: 12px;">5-Year Projection</h3>
            <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">Annual growth: +3.0%</div>
            <table class="event-table">
                <thead>
                    <tr>
                        <th>Year</th>
                        <th>Projected</th>
                        <th>YoY Δ</th>
                    </tr>
                </thead>
                <tbody>
        """, unsafe_allow_html=True)
        
        projection_years = sorted(df[df['is_projection']]['Year'].unique())
        for year in projection_years[:5]:  # Show next 5 years
            proj_data = df[(df['Country'] == country) & (df['Year'] == year)]
            if not proj_data.empty:
                st.markdown(f"""
                <tr class="fade-in">
                    <td>{year}</td>
                    <td>${proj_data['Spending (USD)'].values[0]/1e9:,.1f}B</td>
                    <td class="positive-change">+3.0%</td>
                </tr>
                """, unsafe_allow_html=True)
        
        st.markdown("""
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.divider()
st.caption("""
Data Sources: SIPRI Military Expenditure Database • NATO Annual Reports • Projections based on 3% annual growth model
""")

# Add animations
html("""
<script>
// Simple fade-in animation for elements
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