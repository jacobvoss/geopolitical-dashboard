import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit.components.v1 import html as st_html

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
    }}
    .event-table th,
    .event-table td {{
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    }}
    .event-table th {{
        font-weight: 500;
        color: #94a3b8;
    }}
    .positive-change {{ color: var(--positive); }}
    .negative-change {{ color: var(--negative); }}
    .fade-in {{
        opacity: 0;
        transition: opacity 0.6s ease-out;
    }}
    .fade-in.visible {{
        opacity: 1;
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

    if source == "SIPRI":
        df_melted['YoY_Change'] = df_melted.groupby('Country')['Spending'].pct_change() * 100
        df_melted.rename(columns={'Spending': 'Spending (USD)'}, inplace=True)
    else:
        df_melted.rename(columns={'Spending': 'Spending (% of GDP)'}, inplace=True)

    return df_melted.dropna()

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
        current = country_data.at[event_idx, 'Spending (% of GDP)']
        previous = country_data.at[event_idx - 1, 'Spending (% of GDP)']
        change = current - previous
    else:
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
st.title("Defense Spending Analytics")
st.caption("NATO military expenditure trends with event impact analysis")

with st.sidebar:
    st.header("Filters")
    data_source = st.selectbox("Select Data Source", ["SIPRI", "NATO"])
    df = load_data(data_source)
    available_countries = df['Country'].unique().tolist()
    default_index = available_countries.index('United States') if 'United States' in available_countries else 0
    country = st.selectbox("Select Country", available_countries, index=default_index)
    compare_countries = st.multiselect("Compare With", [c for c in available_countries if c != country])

col1, col2 = st.columns([3, 1])

with col1:
    fig = go.Figure()
    primary_data = df[df['Country'] == country]
    y_col = 'Spending (USD)' if data_source == 'SIPRI' else 'Spending (% of GDP)'

    if data_source == 'SIPRI':
        fig.add_trace(go.Scatter(x=primary_data['Year'], y=primary_data[y_col], name=country,
                                 line=dict(color='#00d2d3', width=3), mode='lines+markers',
                                 hovertemplate="<b>%{x}</b><br>$%{y:,.0f}M<extra></extra>"))
        for c in compare_countries:
            comp_data = df[df['Country'] == c]
            fig.add_trace(go.Scatter(x=comp_data['Year'], y=comp_data[y_col], name=c,
                                     line=dict(dash='dot', width=2), mode='lines'))
        y_title = "Military Spending (USD Millions)"
    else:
        fig.add_trace(go.Bar(x=primary_data['Year'], y=primary_data[y_col], name=country, marker_color='#00d2d3'))
        for c in compare_countries:
            comp_data = df[df['Country'] == c]
            fig.add_trace(go.Bar(x=comp_data['Year'], y=comp_data[y_col], name=c, opacity=0.7))
        min_year = primary_data['Year'].min()
        max_year = primary_data['Year'].max()
        fig.add_shape(type="line", x0=min_year, x1=max_year, y0=2.0, y1=2.0,
                      line=dict(color="#ff6b4a", dash="dash"))
        fig.add_annotation(x=max_year, y=2.0, text="NATO 2% Target", showarrow=False,
                           yshift=10, font=dict(color="#ff6b4a"))
        y_title = "Military Spending (% of GDP)"

    fig.update_layout(title=f"{country} Defense Spending",
                      xaxis_title="Year", yaxis_title=y_title,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#e2e8f0', family="Inter"), height=500,
                      margin=dict(t=80), legend=dict(orientation="h", y=1.02, x=1))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    spending_col = 'Spending (USD)' if data_source == 'SIPRI' else 'Spending (% of GDP)'
    country_data = df[df['Country'] == country].sort_values('Year')
    non_zero_data = country_data[country_data[spending_col] > 0]

    # === Current Spending ===
    if not non_zero_data.empty:
        latest_data = non_zero_data.iloc[-1]
        latest_year = latest_data['Year']
        current_spending = latest_data[spending_col]
        display_value = f"${current_spending/1000:,.1f}B" if data_source == 'SIPRI' else f"{current_spending:.2f}% of GDP"
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div style="color: #94a3b8;">Current Spending</div>
            <div style="font-size: 1.5rem; color: #00d2d3;">{display_value}</div>
            <div style="color: #94a3b8;">{latest_year}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div style="color: #94a3b8;">Current Spending</div>
            <div style="font-size: 1.2rem; color: #ff6b4a;">Data unavailable</div>
        </div>
        """, unsafe_allow_html=True)

    # === 5-Year Change ===
    if not non_zero_data.empty:
        latest_year = non_zero_data.iloc[-1]['Year']
        current_spending = non_zero_data.iloc[-1][spending_col]
        past_year = latest_year - 5
        past_data = country_data[country_data['Year'] == past_year]
        if not past_data.empty:
            past_value = past_data.iloc[0][spending_col]
            if data_source == 'NATO':
                change_str = f"{(current_spending - past_value):+.2f}pp"
                color = "var(--positive)" if current_spending - past_value >= 0 else "var(--negative)"
            else:
                pct_change = (current_spending - past_value) / past_value * 100
                change_str = f"{pct_change:+.1f}%"
                color = "var(--positive)" if pct_change >= 0 else "var(--negative)"
            st.markdown(f"""
            <div class="metric-card fade-in">
                <div style="color: #94a3b8;">5-Year Change</div>
                <div style="font-size: 1.5rem; color: {color};">{change_str}</div>
                <div style="color: #94a3b8;">{past_year} → {latest_year}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card fade-in">
                <div style="color: #94a3b8;">5-Year Change</div>
                <div style="font-size: 1.2rem; color: #ff6b4a;">No data for {past_year}</div>
            </div>
            """, unsafe_allow_html=True)

    # === Event Table ===
    table_html = """
    <div class="fade-in">
        <h3 style="margin-top: 24px;">Key Events</h3>
        <table class="event-table">
            <thead>
                <tr><th>Year</th><th>Event</th><th style="text-align: right;">Impact</th></tr>
            </thead><tbody>
    """
    impacts = [calculate_event_impact(country, y, df) for y in EVENTS["Global"]]
    impacts = [i for i in impacts if i]
    impacts.sort(key=lambda x: abs(x['change']), reverse=True)

    if impacts:
        for i in impacts:
            cls = "positive-change" if i['change'] >= 0 else "negative-change"
            change_str = f"{i['change']:+.2f}pp" if i['is_nato'] else f"{i['change']:+.1f}%"
            table_html += f"<tr class='fade-in'><td>{i['year']}</td><td>{i['name']}</td><td class='{cls}' style='text-align:right'>{change_str}</td></tr>"
    else:
        table_html += "<tr><td colspan='3' style='text-align:center; color:#94a3b8;'>No event impact data available</td></tr>"

    table_html += "</tbody></table></div>"
    st.markdown(table_html, unsafe_allow_html=True)


st.divider()
st.caption("Data Sources: SIPRI Military Expenditure Database • NATO Annual Reports")

# ===== ANIMATIONS =====
st_html("""
<script>
window.addEventListener('load', () => {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, {threshold: 0.1});

    setTimeout(() => {
        document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));
    }, 100);
});
</script>
""", height=0)