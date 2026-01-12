import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
WINDOW_SIZE = 60

MODEL_PATHS = {
    1: "rnn_model_1d.keras",
    5: "rnn_model_5d.keras",
    10: "rnn_model_10d.keras"
}

DATA_PATH = "TSLA.csv"

# Superset of all features ever used
ALL_FEATURES = [
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
# UI
# --------------------------------------------------
st.set_page_config(
    page_title="Tesla Stock Price Prediction (SimpleRNN)",
    layout="wide"
)

st.title("📈 Tesla Stock Price Prediction using SimpleRNN")
st.write(
    "Python 3.13–compatible deployment using trained SimpleRNN models "
    "for 1-day, 5-day, and 10-day forecasting."
)

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
# 🔑 ALIGN FEATURES WITH MODEL
# --------------------------------------------------
expected_features = model.input_shape[2]
FEATURE_COLUMNS = ALL_FEATURES[:expected_features]

st.sidebar.info(
    f"Model expects {expected_features} feature(s): {FEATURE_COLUMNS}"
)

# --------------------------------------------------
# PREPROCESSING (NO JOBLIB)
# --------------------------------------------------
X = df[FEATURE_COLUMNS].copy()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_input = X_scaled[-WINDOW_SIZE:]
X_input = X_input.reshape(
    1,
    WINDOW_SIZE,
    expected_features
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🚀 Predict Closing Price"):
    with st.spinner("Running SimpleRNN prediction..."):

        # Predict (scaled)
        y_pred_scaled = model.predict(X_input, verbose=0)
        # y_pred_scaled shape:
        # (1, 1)  -> 1-day
        # (1, 5)  -> 5-day
        # (1, 10) -> 10-day

        predictions = []

        for i in range(y_pred_scaled.shape[1]):
            dummy = np.zeros((1, expected_features))
            dummy[0, 0] = y_pred_scaled[0, i]  # Close always index 0
            inv = scaler.inverse_transform(dummy)[0, 0]
            predictions.append(inv)

        y_pred = np.array(predictions)

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------
    st.subheader(f"📊 {horizon}-Day Prediction")

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
    "SimpleRNN | Multi-Step Forecasting | Python 3.13 | Streamlit Cloud"
)
