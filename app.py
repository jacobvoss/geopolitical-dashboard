import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Geopolitical Risk Dashboard")

# Load the data
df = pd.read_csv('cleaned_data/cleaned_nato_spending.csv')

# Melt the dataframe to convert years into rows
df_melted = df.melt(id_vars=['Country'],
                    var_name='Year',
                    value_name='Military spending ($USD)')

# Convert Year to numeric
df_melted['Year'] = df_melted['Year'].astype(int)

# Remove rows with missing values
df_melted = df_melted.dropna()

# Country selector
country = st.selectbox("Choose a country", df_melted['Country'].unique())

# Filter data for selected country
filtered = df_melted[df_melted['Country'] == country]

# Create and display the plot
fig = px.line(filtered,
              x='Year',
              y='Military spending ($USD)',
              title=f'{country} Defense Spending Over Time')
st.plotly_chart(fig)