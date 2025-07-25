import pandas as pd
import streamlit as st

@st.cache_data
def load_data(source: str = "SIPRI") -> pd.DataFrame:
    """Load and preprocess defense spending data."""
    if source == "SIPRI":
        df = pd.read_csv('cleaned_data/SIPRI_spending_clean.csv')
    else:
        df = pd.read_csv('cleaned_data/nato_defense_spending_clean.csv')

    df_melted = df.melt(id_vars=['Country'], var_name='Year', value_name='Spending')
    df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
    df_melted = df_melted.dropna(subset=['Year'])
    df_melted['Year'] = df_melted['Year'].astype(int)

    # Only calculate YoY changes for SIPRI (not for NATO)
    if source == "SIPRI":
        df_melted['YoY_Change'] = df_melted.groupby('Country')['Spending'].pct_change() * 100
        df_melted.rename(columns={'Spending': 'Spending (USD)'}, inplace=True)
    else:
        df_melted.rename(columns={'Spending': 'Spending (% of GDP)'}, inplace=True)

    return df_melted.dropna()
