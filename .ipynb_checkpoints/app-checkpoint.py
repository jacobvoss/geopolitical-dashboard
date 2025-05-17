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
    
    .event-annotation {{
        background-color: rgba(255,107,74,0.2);
        padding: 4px 8px;
        border-radius: 4px;
        border-left: 3px solid var(--secondary);
        margin: 8px 0;
    }}
    
    .event-table {{
        width: 100%;
        border-collapse: collapse;
    }}
    .event-table td {{
        padding: 6px;
        border-bottom: 1px solid #334155;
    }}
    .event-table tr:last-child td {{
        border-bottom: none;
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
    
    # Calculate year-over-year changes
    df_melted = df_melted.sort_values(['Country', 'Year'])
    df_melted['YoY_Change'] = df_melted.groupby('Country')['Spending (USD)'].pct_change() * 100
    
    return df_melted.dropna()

df = load_data()
latest_year = df['Year'].max()
available_countries = df['Country'].unique().tolist()

# Historical events data - now with display priority
EVENTS = {
    "Global": {
        2001: {"name": "9/11 Attacks", "priority": 1},
        2008: {"name": "Global Financial Crisis", "priority": 2},
        2014: {"name": "Crimea Annexation", "priority": 1},
        2020: {"name": "COVID-19 Pandemic", "priority": 2},
        2022: {"name": "Russia Invades Ukraine", "priority": 1}
    },
    "United States": {
        2003: {"name": "Iraq War", "priority": 1},
        2011: {"name": "Bin Laden Killed", "priority": 3},
        2017: {"name": "Trump Defense Boost", "priority": 2}
    },
    "Germany": {
        2011: {"name": "Military Reform", "priority": 2},
        2016: {"name": "Defense Spending Increase", "priority": 2}
    }
}

# ===== UI =====
st.title("NATO Defense Spending Analysis")
st.caption("Comparative military expenditure trends with event impact analysis")

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
    priority_filter = st.slider("Event importance filter", 1, 3, 1, 
                              help="Higher numbers show more events")

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
        mode='lines',
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>"
    ))
    
    # Comparison countries
    for c in compare_countries:
        comp_data = df[df['Country'] == c]
        fig.add_trace(go.Scatter(
            x=comp_data['Year'],
            y=comp_data['Spending (USD)'],
            name=c,
            line=dict(width=1.5, dash='dot'),
            opacity=0.7,
            hovertemplate="<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>"
        ))
    
    # Add event annotations if enabled
    if show_events:
        # Combine global and country-specific events
        all_events = {**EVENTS.get("Global", {}), **EVENTS.get(country, {})}
        
        # Filter by priority
        filtered_events = {yr: ev for yr, ev in all_events.items() 
                          if ev["priority"] <= priority_filter and yr in primary_data['Year'].values}
        
        # Calculate positions to avoid overlap
        positions = ["top", "bottom"] * len(filtered_events)
        
        for i, (year, event) in enumerate(filtered_events.items()):
            # Get spending change for this event year
            event_data = primary_data[primary_data['Year'] == year]
            prev_year_data = primary_data[primary_data['Year'] == year - 1]
            
            if not prev_year_data.empty:
                yoy_change = event_data['YoY_Change'].values[0]
                change_text = f"{'+' if yoy_change >= 0 else ''}{yoy_change:.1f}%"
                
                fig.add_vline(
                    x=year,
                    line_width=1,
                    line_dash="dash",
                    line_color="#ff6b4a",
                    opacity=0.5,
                    annotation_text=f"{event['name']}<br>{change_text}",
                    annotation_position=positions[i % 2],
                    annotation_font_size=10,
                    annotation_bgcolor="rgba(30,41,59,0.8)"
                )
    
    fig.update_layout(
        title=f"{country} Defense Spending Over Time",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        hovermode="x unified",
        height=500,
        margin=dict(t=80)  # Extra space for annotations
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Get latest non-zero spending value (fixes 2024 issue)
    country_data = df[df['Country'] == country]
    latest_non_zero = country_data[country_data['Spending (USD)'] > 0].sort_values('Year').iloc[-1]
    
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
    
    # Event explanations table
    if show_events:
        st.markdown("### Key Events & Impact")
        
        # Combine and filter events
        all_events = {**EVENTS.get("Global", {}), **EVENTS.get(country, {})}
        filtered_events = {yr: ev for yr, ev in all_events.items() 
                         if ev["priority"] <= priority_filter and yr in primary_data['Year'].values}
        
        # Create event table
        event_table = "<table class='event-table'><tr><th>Year</th><th>Event</th><th>YoY Δ</th></tr>"
        
        for year, event in sorted(filtered_events.items()):
            event_data = primary_data[primary_data['Year'] == year]
            prev_year_data = primary_data[primary_data['Year'] == year - 1]
            
            if not prev_year_data.empty:
                yoy_change = event_data['YoY_Change'].values[0]
                change_text = f"{'+' if yoy_change >= 0 else ''}{yoy_change:.1f}%"
                change_color = "#10b981" if yoy_change > 0 else "#ff6b4a"
                
                event_table += f"""
                <tr>
                    <td>{year}</td>
                    <td>{event['name']}</td>
                    <td style='color: {change_color}'>{change_text}</td>
                </tr>
                """
        
        event_table += "</table>"
        st.markdown(event_table, unsafe_allow_html=True)

# Footer
st.divider()
st.caption("Data Sources: SIPRI Military Expenditure Database • NATO Annual Reports")