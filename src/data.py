import pandas as pd

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


def load_data(source: str = "SIPRI") -> pd.DataFrame:
    """Load and preprocess spending data from the cleaned CSV files."""
    if source == "SIPRI":
        df = pd.read_csv("cleaned_data/SIPRI_spending_clean.csv")
    else:
        df = pd.read_csv("cleaned_data/nato_defense_spending_clean.csv")

    df_melted = df.melt(id_vars=["Country"], var_name="Year", value_name="Spending")
    df_melted["Year"] = pd.to_numeric(df_melted["Year"], errors="coerce")
    df_melted = df_melted.dropna(subset=["Year"])
    df_melted["Year"] = df_melted["Year"].astype(int)

    if source == "SIPRI":
        df_melted["YoY_Change"] = df_melted.groupby("Country")["Spending"].pct_change() * 100
        df_melted.rename(columns={"Spending": "Spending (USD)"}, inplace=True)
    else:
        df_melted.rename(columns={"Spending": "Spending (% of GDP)"}, inplace=True)

    return df_melted.dropna()


def calculate_event_impact(country: str, year: int, df: pd.DataFrame):
    """Return impact information for an event affecting the specified country."""
    is_nato = "Spending (% of GDP)" in df.columns
    country_data = df[df["Country"] == country].sort_values("Year").reset_index(drop=True)

    try:
        event_idx = country_data.index[country_data["Year"] == year][0]
    except IndexError:
        return None

    if event_idx == 0:
        return None

    if is_nato:
        current = country_data.at[event_idx, "Spending (% of GDP)"]
        previous = country_data.at[event_idx - 1, "Spending (% of GDP)"]
        change = current - previous
    else:
        current = country_data.at[event_idx, "Spending (USD)"]
        previous = country_data.at[event_idx - 1, "Spending (USD)"]
        change = (current - previous) / previous * 100

    return {
        "year": year,
        "name": EVENTS.get("Global", {}).get(year),
        "change": change,
        "prev_year": country_data.at[event_idx - 1, "Year"],
        "is_nato": is_nato,
    }
