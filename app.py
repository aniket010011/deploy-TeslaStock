import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# --------------------------------------------------
# CONFIG (MUST MATCH TRAINING)
# --------------------------------------------------
WINDOW_SIZE = 60

MODEL_PATHS = {
    1: "rnn_model_1d.keras",
    5: "rnn_model_5d.keras",
    10: "rnn_model_10d.keras"
}

DATA_PATH = "TSLA.csv"

# EXACT feature order used during training
FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "MA20",
    "MA50",
    "Rolling_STD_20"
]

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # Feature engineering (same as notebook)
    df["Close"] = df["Close"].interpolate(method="time")
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Rolling_STD_20"] = df["Close"].rolling(20).std()

    df.dropna(inplace=True)
    return df

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_rnn_model(horizon):
    return load_model(MODEL_PATHS[horizon])

# --------------------------------------------------
# UI SETUP
# --------------------------------------------------
st.set_page_config(
    page_title="Tesla Stock Price Prediction (SimpleRNN)",
    layout="wide"
)

st.title("📈 Tesla Stock Price Prediction using SimpleRNN")
st.write(
    "Python 3.13–compatible deployment using trained **SimpleRNN** models "
    "for 1-day, 5-day, and 10-day forecasting."
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Prediction Settings")

horizon = st.sidebar.selectbox(
    "Prediction Horizon (Days Ahead)",
    [1, 5, 10]
)

model = load_rnn_model(horizon)

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------
with st.expander("🔍 View Latest Data"):
    st.dataframe(df.tail(10))

# --------------------------------------------------
# PREPROCESSING (NO JOBLIB, NO PICKLE)
# --------------------------------------------------
X = df[FEATURE_COLUMNS].copy()

# Fit scaler on historical data (deployment-safe)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Build input window
X_input = X_scaled[-WINDOW_SIZE:]

# Enforce exact model input shape
X_input = X_input.reshape(
    1,
    WINDOW_SIZE,
    len(FEATURE_COLUMNS)
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🚀 Predict Closing Price"):
    with st.spinner("Running SimpleRNN prediction..."):

        # Predict (scaled)
        y_pred_scaled = model.predict(X_input, verbose=0)

        # Inverse scale ONLY the Close price
        dummy = np.zeros((y_pred_scaled.shape[0], len(FEATURE_COLUMNS)))
        dummy[:, 0] = y_pred_scaled.flatten()  # Close is index 0

        y_pred = scaler.inverse_transform(dummy)[:, 0]

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------
    st.subheader(f"📊 {horizon}-Day Prediction (SimpleRNN)")

    if horizon == 1:
        st.metric(
            label="Predicted Closing Price",
            value=f"${y_pred[0]:.2f}"
        )
    else:
        st.dataframe(
            pd.DataFrame(y_pred, columns=["Predicted Close"])
        )

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------
    st.subheader("📉 Actual vs Predicted Trend")

    recent_actual = df[["Close"]].tail(100)

    future_dates = pd.date_range(
        start=recent_actual.index[-1],
        periods=horizon + 1,
        freq="B"
    )[1:]

    future_pred = pd.DataFrame(
        y_pred,
        index=future_dates,
        columns=["Close"]
    )

    combined = pd.concat([recent_actual, future_pred])

    st.line_chart(combined)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
    "SimpleRNN | Python 3.13 Safe | Streamlit Cloud Deployment"
)
