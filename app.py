import streamlit as st
import pandas as pd
import numpy as np
import pickle, gzip
import plotly.express as px

import pickle, gzip

def load_model():
    with gzip.open("covid_model_small.pkl.gz", "rb") as f:
        return pickle.load(f)

# ---------------------------
# 1. Page Config
# ---------------------------
st.set_page_config(
    page_title="COVID-19 5-Year Prediction",
    page_icon="🦠",
    layout="wide"
)

st.title("🦠 COVID-19 Case Prediction Dashboard")
st.markdown("### Predict the next 5 years of COVID-19 cases for any country in the dataset.")

# ---------------------------
# 2. Load Data & Model
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("owid-covid-data.csv")
    df = df[['location','date','total_cases','new_cases','total_deaths',
             'population','gdp_per_capita','hospital_beds_per_thousand']]
    df['date'] = pd.to_datetime(df['date'])
    df['new_cases_7d_avg'] = df.groupby('location')['new_cases'].transform(lambda x: x.rolling(7).mean())
    df['cases_growth_rate'] = df.groupby('location')['total_cases'].pct_change().fillna(0)
    df = df.fillna(0)
    return df

@st.cache_resource
def load_model():
    with gzip.open("covid_model_small.pkl.gz", "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

features = [
    'total_cases','total_deaths','population','gdp_per_capita',
    'hospital_beds_per_thousand','new_cases_7d_avg','cases_growth_rate'
]

# ---------------------------
# 3. Sidebar - User Controls
# ---------------------------
st.sidebar.header("⚙️ Options")
country = st.sidebar.selectbox("Select Country/Region", sorted(df['location'].unique()))
future_years = st.sidebar.slider("Prediction Years", 1, 5, 5)

country_df = df[df['location']==country].sort_values('date')

# Show historical chart
st.subheader(f"📊 Historical Daily New Cases for {country}")
fig_hist = px.line(
    country_df, x='date', y='new_cases',
    title=f"{country} - Historical New Cases",
    labels={"new_cases": "New Cases", "date": "Date"}
)
st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------
# 4. Predict Future Cases
# ---------------------------
if st.button(f"🔮 Predict Next {future_years} Years"):
    future_days = 365 * future_years
    last_known = country_df.iloc[-1:].copy()
    future_preds = []

    for i in range(future_days):
        X_last = last_known[features]
        pred = model.predict(X_last)[0]
        future_preds.append(pred)

        # Prepare next day
        new_row = last_known.copy()
        new_row['date'] = new_row['date'] + pd.Timedelta(days=1)
        new_row['new_cases'] = pred
        new_row['total_cases'] = new_row['total_cases'] + pred
        new_row['new_cases_7d_avg'] = (new_row['new_cases_7d_avg']*6 + pred)/7
        new_row['cases_growth_rate'] = pred / max(new_row['total_cases'].values[0], 1)
        last_known = new_row

    # Prepare prediction DataFrame
    future_dates = pd.date_range(
        start=country_df['date'].iloc[-1] + pd.Timedelta(days=1),
        periods=future_days
    )
    future_df = pd.DataFrame({
        "date": future_dates,
        "predicted_new_cases": future_preds
    })

    st.subheader(f"📈 {country} - Predicted New Cases for Next {future_years} Years")
    fig_pred = px.line(
        future_df, x='date', y='predicted_new_cases',
        title=f"{country} - 5-Year Predicted Daily New Cases",
        labels={"predicted_new_cases": "Predicted New Cases", "date": "Date"},
        line_shape='spline'
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Download CSV
    csv = future_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name=f"{country}_future_covid_predictions.csv",
        mime='text/csv'
    )

    st.success("✅ Prediction complete and ready to download!")

# ---------------------------
# 5. Extra Info
# ---------------------------
st.sidebar.info("This app predicts daily new COVID-19 cases for up to 5 years using a lightweight RandomForest model.")
st.sidebar.markdown("**Tip:** Use the download button to get CSV results.")

import os
import gzip
import pickle

@st.cache_data
def load_model():
    model_path = os.path.join(os.getcwd(), "covid_model_small.pkl.gz")  # Adjust if needed
    with gzip.open(model_path, "rb") as f:
        return pickle.load(f)

