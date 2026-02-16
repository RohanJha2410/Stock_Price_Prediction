import streamlit as st
import yfinance as yf
import numpy as np
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="LSTM Stock Forecast",
    layout="wide",
    page_icon="📈"
)

st.markdown("""
# 📈 LSTM Stock Price Forecast Dashboard
Predict the **next 30 days Open price** using a trained LSTM model.
""")

st.markdown("---")

# ------------------------------------------------
# Load Model & Scaler
# ------------------------------------------------
@st.cache_resource
def load_assets():
    model = load_model("lstm_model.h5")
    scaler = joblib.load("scaler.save")
    return model, scaler

model, scaler = load_assets()

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.header("⚙️ Settings")
stock_symbol = st.sidebar.text_input("Stock Symbol", "GAIL.NS")
predict_button = st.sidebar.button("🚀 Predict")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info:**")
st.sidebar.write("• Architecture: 2-Layer LSTM")
st.sidebar.write("• Input Window: 100 Days")
st.sidebar.write("• Forecast Horizon: 30 Days")
st.sidebar.write("• Data Source: Yahoo Finance")

# ------------------------------------------------
# Main Prediction Logic
# ------------------------------------------------
if predict_button:

    with st.spinner("Downloading data and generating forecast..."):

        data = yf.download(stock_symbol, period="10y", interval="1d")

        if data.empty:
            st.error("Invalid stock symbol or no data found.")
            st.stop()

        opn = data[['Open']]
        ds = opn.values
        ds_scaled = scaler.transform(ds)

        # Last 100 days
        last_100 = ds_scaled[-100:]
        tmp_inp = last_100.reshape(1, -1).tolist()[0]

        lst_output = []
        n_steps = 100

        # Recursive forecast
        for i in range(30):
            x_input = np.array(tmp_inp[-100:]).reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            tmp_inp.append(yhat[0][0])
            lst_output.append(yhat[0][0])

        forecast_array = np.array(lst_output).reshape(-1,1)
        ds_new = np.vstack((ds_scaled, forecast_array))
        final_graph = scaler.inverse_transform(ds_new)

    # ------------------------------------------------
    # KPI Metrics Section
    # ------------------------------------------------
    latest_price = float(opn.values[-1][0])

    predicted_price = float(final_graph[-1][0])
    price_change = predicted_price - latest_price
    percent_change = (price_change / latest_price) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("📍 Current Price", f"₹ {round(latest_price,2)}")
    col2.metric("🔮 Predicted (30D)", f"₹ {round(predicted_price,2)}")
    col3.metric(
        "📊 Expected Change",
        f"{round(price_change,2)}",
        f"{round(percent_change,2)} %"
    )

    st.markdown("---")

    # ------------------------------------------------
    # Interactive Plot
    # ------------------------------------------------
    import pandas as pd

    # Historical dates
    historical_dates = data.index

    # Create future 30 business dates
    future_dates = pd.date_range(
        start=historical_dates[-1] + pd.Timedelta(days=1),
        periods=30,
        freq='B'  # Business days
    )

    # Combine dates
    all_dates = historical_dates.append(future_dates)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=all_dates,
        y=final_graph.flatten(),
        mode='lines',
        name='Historical + Forecast'
    ))

   # Vertical Forecast Start Line (SAFE)
    fig.add_shape(
        type="line",
        x0=str(historical_dates[-1]),
        x1=str(historical_dates[-1]),
        y0=float(final_graph.min()),
        y1=float(final_graph.max()),
        line=dict(color="orange", dash="dash")
    )

    fig.add_annotation(
        x=str(historical_dates[-1]),
        y=float(final_graph.max()),
        text="Forecast Start",
        showarrow=False,
        yshift=10
    )

    # Horizontal Target Line (SAFE)
    fig.add_shape(
        type="line",
        x0=str(all_dates[0]),
        x1=str(all_dates[-1]),
        y0=predicted_price,
        y1=predicted_price,
        line=dict(color="red", dash="dot")
    )

    fig.add_annotation(
        x=str(all_dates[-1]),
        y=predicted_price,
        text=f"30D Target: ₹ {round(predicted_price,2)}",
        showarrow=False,
        yshift=10
    )



    fig.update_layout(
        title=f"{stock_symbol} - 30 Day LSTM Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=600
    )




    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------
    # Extra Info
    # ------------------------------------------------
    st.markdown("---")
    st.caption(f"Last Updated: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}")

