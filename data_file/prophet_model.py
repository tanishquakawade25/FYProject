import yfinance as yf
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ParameterGrid
import streamlit as st

def validate_ticker(ticker):
    """
    Validates if the stock ticker exists and can be fetched using yfinance.
    
    Parameters:
    - ticker (str): Stock ticker symbol (e.g., 'AAPL').
    
    Returns:
    - bool: True if ticker is valid, False otherwise.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")  # Attempt to fetch a single day's data to check validity
        if data.empty:
            return False
        return True
    except Exception:
        return False

def run_prophet_model(manual_stocks, start_date, end_date, prediction_mode="Tabular"):
    """
    Runs a Prophet model for stock price prediction using the specified stock ticker and date range.
    
    Parameters:
    - manual_stocks (str): Stock ticker symbol (e.g., 'BSOFT.NS').
    - start_date (str): Start date for historical data (YYYY-MM-DD).
    - end_date (str): End date for historical data (YYYY-MM-DD).
    - prediction_mode (str): Display mode for predictions ('Tabular' or 'Visual').
    """
    st.subheader("Stock Price Predictions using Prophet Model")

    # Check if the ticker is empty
    if not manual_stocks:
        st.warning("No valid tickers found. Please check your input or data availability.")
        return

    # Validate stock ticker
    if not validate_ticker(manual_stocks):
        st.error(f"Invalid stock ticker: {manual_stocks}. Please enter a valid ticker.")
        return

    prediction_mode = st.radio(
        "Choose Visualisation Mode : ",
        ("Tabular", "Graphical"),
        horizontal=True
    )

    # Fetch stock data
    @st.cache_data
    def fetch_hourly_data_with_period(ticker, start_date, end_date):
        all_data = []
        current_start = pd.Timestamp(start_date)
        final_end = pd.Timestamp(end_date)

        while current_start < final_end:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(interval='1h', period="1d", 
                                     start=current_start.date(), 
                                     end=(current_start + pd.Timedelta(days=1)).date())
                if not data.empty:
                    all_data.append(data)
            except Exception as e:
                st.write(f"Error fetching data for {current_start.date()}: {e}")

            current_start += pd.Timedelta(days=1)

        if all_data:
            combined_data = pd.concat(all_data)
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
            return combined_data
        else:
            raise ValueError("No data fetched. Verify the ticker symbol or date range.")

    try:
        stock_data = fetch_hourly_data_with_period(manual_stocks, start_date, end_date)
        vix_data = fetch_hourly_data_with_period('^INDIAVIX', start_date, end_date)
    except ValueError as e:
        st.error(f"Error: {e}")
        return

    # Process and merge data
    stock_data['Date'] = stock_data.index.date
    stock_data = stock_data.reset_index().rename(columns={'Close': 'Stock Price'})
    stock_data = stock_data[['Date', 'Stock Price']]

    vix_data['Date'] = vix_data.index.date
    vix_data = vix_data.reset_index().rename(columns={'Close': 'VIX'})
    vix_data = vix_data[['Date', 'VIX']]

    merged_df = pd.merge(stock_data, vix_data, on='Date', how='inner')
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])

    # Prepare data for Prophet
    df_prophet = merged_df.rename(columns={'Date': 'ds', 'Stock Price': 'y', 'VIX': 'vix'})
    df_prophet['vix'] = df_prophet['vix'].fillna(method='ffill')

    # Hyperparameter tuning
    param_grid = {
        'seasonality_prior_scale': [10, 15, 20],
        'changepoint_prior_scale': [0.05, 0.1, 0.2],
    }

    best_model = None
    best_score = float('inf')
    best_params = None

    for params in ParameterGrid(param_grid):
        model = Prophet(**params, yearly_seasonality=True)
        model.add_seasonality(name='weekly', period=7, fourier_order=3)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_regressor('vix')
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=0, freq='D', include_history=True)
        future['vix'] = df_prophet['vix']
        forecast = model.predict(future)

        merged = pd.merge(forecast[['ds', 'yhat']], df_prophet[['ds', 'y']], on='ds', how='inner')
        y_pred = merged['yhat']
        y_true = merged['y']

        if len(y_true) == len(y_pred):
            mape = mean_absolute_percentage_error(y_true, y_pred)

            if mape < best_score:
                best_score = mape
                best_params = params

    best_model = Prophet(**best_params, yearly_seasonality=True)
    best_model.add_seasonality(name='weekly', period=7, fourier_order=3)
    best_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    best_model.add_regressor('vix')
    best_model.fit(df_prophet)

    # Forecast future
    future = best_model.make_future_dataframe(periods=5, freq='D', include_history=True)
    future['vix'] = df_prophet['vix'].iloc[-1]
    forecast = best_model.predict(future)

    # Calculate evaluation metrics
    merged = pd.merge(forecast[['ds', 'yhat']], df_prophet[['ds', 'y']], on='ds', how='inner')
    y_pred = merged['yhat']
    y_true = merged['y']

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Prepare future dataframe for visualization
    future_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
        'ds': 'Date', 'yhat': 'Predicted Close', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'
    })

    # Visualization Mode
    if prediction_mode == "Tabular":
        # Display tabular data
        st.subheader("Tabular Data")
        st.dataframe(future_df.tail(), use_container_width=True)

        # Save tabular data to CSV
        csv_data = future_df.to_csv(index=False)
        st.download_button(
            label="Download Predicted Data as CSV",
            data=csv_data,
            file_name=f"{manual_stocks}_predicted_data.csv",
            mime="text/csv",
            icon="⬇️",
        )

    elif prediction_mode == "Graphical":
        # Plot historical and predicted data
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot historical data
        ax1.plot(
            df_prophet['ds'], 
            df_prophet['y'], 
            label="Historical Prices", 
            color="blue", 
            linewidth=2
        )

        # Plot predicted data
        ax1.plot(
            forecast['ds'], 
            forecast['yhat'], 
            label="Predicted Prices", 
            color="green", 
            linestyle="--"
        )

        # Add confidence interval shading
        ax1.fill_between(
            forecast['ds'], 
            forecast['yhat_lower'], 
            forecast['yhat_upper'], 
            color="green", 
            alpha=0.2, 
            label="Confidence Interval"
        )

        # Set plot titles and labels
        ax1.set_title(f"{manual_stocks} Close Price: Historical and Forecast", fontsize=16, fontweight='bold')
        ax1.set_xlabel("Date", fontsize=14)
        ax1.set_ylabel("Price", fontsize=14)

        # Customize grid and legend
        ax1.legend(fontsize=12, loc="upper left")
        ax1.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

        # Display the plot
        st.pyplot(fig)

        # Save the plot to an in-memory file
        image_path = f"{manual_stocks}_predicted_plot.png"
        fig.savefig(image_path, format="png")

        # Allow user to download the plot as an image
        with open(image_path, "rb") as img_file:
            st.download_button(
                label=f"Download {manual_stocks} Plot as Image",
                data=img_file,
                file_name=image_path,
                mime="image/png",
                icon="📥",
            )

    # Display evaluation metrics
    st.write("\nModel Evaluation (Overall):")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R-squared Score (R²): {r2:.2f}")
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")
