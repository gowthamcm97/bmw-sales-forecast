import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

@st.cache_data
def load_data(uploaded_file):
    """Load uploaded CSV and aggregate monthly."""
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    # Aggregate by month
    monthly = df.groupby(df['Date'].dt.to_period('M')).agg({
        'Units_Sold': 'sum',
        'Avg_Price_EUR': 'mean',
        'BEV_Share': 'mean',
        'Premium_Share': 'mean',
        'GDP_Growth': 'mean',
        'Fuel_Price_Index': 'mean'
    }).reset_index()
    monthly['Date'] = monthly['Date'].dt.to_timestamp()
    monthly = monthly.sort_values('Date').reset_index(drop=True)
    return monthly

def create_features(df, target='Units_Sold'):
    """Add lag, rolling, and seasonal features."""
    df = df.copy()
    # Lags
    df['Sales_Lag_1'] = df[target].shift(1)
    df['Sales_Lag_3'] = df[target].shift(3)
    df['Sales_Lag_6'] = df[target].shift(6)
    # Rolling means
    df['Sales_Rolling_3'] = df[target].rolling(3).mean()
    df['Sales_Rolling_6'] = df[target].rolling(6).mean()
    # Growth rate
    df['Sales_Growth'] = df[target].pct_change()
    # Calendar features
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    # Interaction
    df['EV_Demand_Index'] = df['BEV_Share'] * df['Fuel_Price_Index']
    return df

@st.cache_resource
def load_model_scaler():
    model = joblib.load('models/bmw_sales_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    return model, scaler, feature_cols
def recursive_forecast(model, scaler, feature_cols, last_row, monthly_features, periods=12):
    """
    Recursively forecast next `periods` months.
    Exogenous variables are held constant at last observed values.
    """
    future_dates = pd.date_range(
        start=last_row['Date'] + pd.DateOffset(months=1),
        periods=periods,
        freq='MS'
    )
    current_features = monthly_features.copy()
    forecasts = []

    # Expected numeric feature order from scaler
    expected_numeric = list(scaler.feature_names_in_)

    for i in range(periods):
        next_date = future_dates[i]
        # Base row with constant exogenous values
        new_row = {
            'Date': next_date,
            'Avg_Price_EUR': last_row['Avg_Price_EUR'],
            'BEV_Share': last_row['BEV_Share'],
            'Premium_Share': last_row['Premium_Share'],
            'GDP_Growth': last_row['GDP_Growth'],
            'Fuel_Price_Index': last_row['Fuel_Price_Index'],
            'Month': next_date.month,
            'Quarter': next_date.quarter,
        }

        # Compute lags and rolling from the most recent 6 values
        last_6 = current_features['Units_Sold'].iloc[-6:].tolist()
        # Lags
        new_row['Sales_Lag_1'] = last_6[-1] if len(last_6) >= 1 else 0
        new_row['Sales_Lag_3'] = last_6[-3] if len(last_6) >= 3 else 0
        new_row['Sales_Lag_6'] = last_6[-6] if len(last_6) >= 6 else 0
        # Rolling means
        new_row['Sales_Rolling_3'] = np.mean(last_6[-3:]) if len(last_6) >= 3 else np.nan
        new_row['Sales_Rolling_6'] = np.mean(last_6[-6:]) if len(last_6) >= 6 else np.nan
        # Growth rate
        if len(last_6) >= 2:
            new_row['Sales_Growth'] = (last_6[-1] / last_6[-2]) - 1
        else:
            new_row['Sales_Growth'] = 0.0
        # EV demand index
        new_row['EV_Demand_Index'] = new_row['BEV_Share'] * new_row['Fuel_Price_Index']

        # Create DataFrame for this row
        row_df = pd.DataFrame([new_row])

        # Build numeric vector in exact scaler order
        numeric_values = []
        for feat in expected_numeric:
            if feat in row_df.columns:
                numeric_values.append(row_df[feat].iloc[0])
            else:
                numeric_values.append(0.0)  # fallback

        # Scale the numeric vector
        scaled_numeric = scaler.transform([numeric_values])[0]

        # Update row_df with scaled values
        for j, feat in enumerate(expected_numeric):
            row_df[feat] = scaled_numeric[j]

        # Reorder all features to match training order (feature_cols)
        row_df_ordered = row_df[feature_cols]

        # Predict
        drow = xgb.DMatrix(row_df_ordered)
        pred = model.predict(drow)[0]
        forecasts.append(pred)

        # Add the predicted row to current_features for next iteration
        new_row['Units_Sold'] = pred
        current_features = pd.concat([current_features, pd.DataFrame([new_row])], ignore_index=True)

    return future_dates, forecasts

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="BMW Sales Forecast", layout="wide")
st.title("🚗 BMW Monthly Sales Forecast (Next 12 Months)")
st.markdown("Upload your cleaned BMW sales data (CSV) to generate a 12‑month recursive forecast using your trained XGBoost model.")

# Sidebar for file upload
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.markdown("---")
    st.markdown("**Model files required**")
    st.markdown("Make sure `bmw_sales_model.pkl`, `scaler.pkl`, and `feature_cols.pkl` are in the same folder as this app.")

# Main panel
if uploaded_file is not None:
    # Load and aggregate data
    with st.spinner("Loading and aggregating data..."):
        monthly = load_data(uploaded_file)
    st.success(f"Data loaded. Aggregated to {len(monthly)} months.")

    # Feature engineering
    with st.spinner("Creating lag/rolling features..."):
        monthly_features = create_features(monthly)
        # Drop initial NaN rows
        monthly_features = monthly_features.dropna().reset_index(drop=True)
    st.info(f"Feature‑engineered shape: {monthly_features.shape}")

    # Load model
    try:
        model, scaler, feature_cols = load_model_scaler()
        st.success("Model loaded successfully.")
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}. Please ensure they are in the app directory.")
        st.stop()

    # Get last row for forecast
    last_row = monthly_features.iloc[-1].copy()

    # Forecast button
    if st.button("🚀 Run 12‑Month Forecast"):
        with st.spinner("Recursive forecasting in progress..."):
            future_dates, forecasts = recursive_forecast(
                model, scaler, feature_cols, last_row, monthly_features, periods=12
            )

        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Units_Sold': [int(round(x)) for x in forecasts]
        })

        # Display results
        st.subheader("Forecast for the next 12 months")
        st.dataframe(forecast_df, use_container_width=True)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(monthly_features['Date'], monthly_features['Units_Sold'],
                label='Historical', color='blue', linewidth=2)
        ax.plot(forecast_df['Date'], forecast_df['Predicted_Units_Sold'],
                'ro-', label='Forecast', linewidth=2, markersize=6)
        ax.set_xlabel('Date')
        ax.set_ylabel('Units Sold')
        ax.set_title('BMW Monthly Sales Forecast')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Option to download forecast
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast as CSV",
            data=csv,
            file_name='bmw_12month_forecast.csv',
            mime='text/csv'
        )
else:
    st.info("👈 Please upload your CSV file to begin.")

# Instructions in the sidebar (collapsible)
with st.sidebar.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Prepare your data** – The CSV should contain the columns from your cleaned dataset (e.g., `Date`, `Units_Sold`, `Avg_Price_EUR`, `BEV_Share`, …).  
    2. **Upload** the file using the uploader.  
    3. **Click** the forecast button.  
    4. **View** the table and plot, and download the forecast.

    **Assumptions**  
    - External variables (`GDP_Growth`, `Fuel_Price_Index`, etc.) are kept constant at their last observed value.  
    - The model was trained on the same feature set.
    """)