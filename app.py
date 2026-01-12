import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# ----------------------------------------------------
# CONFIG (must match notebook)
# ----------------------------------------------------
WINDOW_SIZE = 60

MODEL_PATHS = {
    1: "rnn_model_1d.keras",
    5: "rnn_model_5d.keras",
    10: "rnn_model_10d.keras"
}

DATA_PATH = "TSLA.csv"
PIPELINE_PATH = "preprocessing_pipeline.joblib"

# ----------------------------------------------------
# LOAD ARTIFACTS
# ----------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)

@st.cache_resource
def load_model_by_horizon(horizon):
    return load_model(MODEL_PATHS[horizon])

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # same preprocessing logic as notebook
    df["Close"] = df["Close"].interpolate(method="time")

    # technical indicators used in notebook
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Rolling_STD_20"] = df["Close"].rolling(20).std()

    df.dropna(inplace=True)
    return df

# ----------------------------------------------------
# UI
# ----------------------------------------------------
st.set_page_config(
    page_title="Tesla Stock Price Prediction (SimpleRNN)",
    layout="wide"
)

st.title("📈 Tesla Stock Price Prediction using SimpleRNN")
st.write(
    "This app uses **trained SimpleRNN models** (1-day, 5-day, 10-day horizons) "
    "based on the final notebook evaluation."
)

# ----------------------------------------------------
# LOAD DATA & PIPELINE
# ----------------------------------------------------
df = load_data()
preprocessor = load_pipeline()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.header("Prediction Settings")

horizon = st.sidebar.selectbox(
    "Prediction Horizon (Days Ahead)",
    options=[1, 5, 10]
)

model = load_model_by_horizon(horizon)

# ----------------------------------------------------
# DATA PREVIEW
# ----------------------------------------------------
with st.expander("🔍 View Latest Data"):
    st.dataframe(df.tail(10))

# ----------------------------------------------------
# PREPARE MODEL INPUT (MATCH NOTEBOOK)
# ----------------------------------------------------
# Feature columns come directly from the trained pipeline
FEATURE_COLUMNS = preprocessor.transformers_[0][2]

X_input_raw = df[FEATURE_COLUMNS].tail(WINDOW_SIZE)

X_input_scaled = preprocessor.transform(X_input_raw)

X_input = X_input_scaled.reshape(
    1,
    WINDOW_SIZE,
    X_input_scaled.shape[1]
)

# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------
if st.button("🚀 Predict Closing Price"):
    with st.spinner("Running SimpleRNN prediction..."):

        y_pred_scaled = model.predict(X_input, verbose=0)

        scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
        y_pred = scaler.inverse_transform(y_pred_scaled)

    # ------------------------------------------------
    # OUTPUT
    # ------------------------------------------------
    st.subheader(f"📊 {horizon}-Day Prediction (SimpleRNN)")

    if horizon == 1:
        st.metric(
            label="Predicted Closing Price",
            value=f"${y_pred[0, 0]:.2f}"
        )
    else:
        pred_df = pd.DataFrame(
            y_pred.flatten(),
            columns=["Predicted Close"]
        )
        st.dataframe(pred_df)

    # ------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------
    st.subheader("📉 Actual vs Predicted Trend")

    recent_actual = df[["Close"]].tail(100)

    future_dates = pd.date_range(
        start=recent_actual.index[-1],
        periods=horizon + 1,
        freq="B"
    )[1:]

    future_pred = pd.DataFrame(
        y_pred.flatten(),
        index=future_dates,
        columns=["Close"]
    )

    combined = pd.concat([recent_actual, future_pred])

    st.line_chart(combined)

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown("---")
st.caption(
    "SimpleRNN-based Time Series Forecasting | "
    "Aligned with Final Notebook Evaluation"
)
