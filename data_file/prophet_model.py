import yfinance as yf 
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st

def run_prophet_model(manual_stocks, start_date, end_date, prediction_mode="Tabular"): 
    
    print("Prophet model is imported")

    ticker_list = [ticker.strip() for ticker in manual_stocks.split(",")]

    # Fetch and process stock data
    def fetch_hourly_data(ticker, start_date, end_date):
        all_data = []
        current_start = pd.Timestamp(start_date)
        final_end = pd.Timestamp(end_date)

        while current_start < final_end:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(
                    interval='1h',
                    period="1d",
                    start=current_start.date(),
                    end=(current_start + pd.Timedelta(days=1)).date()
                )
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
            st.error(f"No data fetched for ticker {ticker}. Verify the symbol or date range.")
            return pd.DataFrame()

    # Fetch data for multiple tickers
    def fetch_and_process_data(tickers, start_date, end_date):
        all_data = []
        for ticker in tickers:
            st.write(f"Fetching data for {ticker}...")
            stock_data = fetch_hourly_data(ticker, start_date, end_date)
            if not stock_data.empty:
                stock_data['Ticker'] = ticker
                all_data.append(stock_data)
            else:
                st.warning(f"No data fetched for {ticker}.")
        if all_data:
            combined_data = pd.concat(all_data)
            return combined_data
        else:
            st.error("No data fetched for any ticker.")
            return pd.DataFrame()

    # Fetch stock data for all tickers
    all_stock_data = fetch_and_process_data(ticker_list, start_date, end_date)

    # Fetch VIX data
    st.write("Fetching VIX data...")
    vix_data = fetch_hourly_data("^INDIAVIX", start_date, end_date)

    if not all_stock_data.empty and not vix_data.empty:
        # Prepare VIX data and merge with stock data
        vix_data['Date'] = vix_data.index.date
        vix_data = vix_data[['Close', 'Date']].rename(columns={'Close': 'VIX'}).reset_index(drop=True)

        # Accumulators for metrics
        all_mse = []
        all_rmse = []
        all_r2 = []
        all_percent_mse = []

        results = {}
        for ticker in ticker_list:
            st.write(f"Processing data for {ticker}...")
            stock_data = all_stock_data[all_stock_data['Ticker'] == ticker]

            # Prepare stock data and merge with VIX
            stock_data['Date'] = stock_data.index.date
            stock_data = stock_data.rename(columns={'Close': 'Stock Price'})
            stock_data = stock_data.reset_index(drop=True)
            merged_df = pd.merge(stock_data, vix_data, on='Date', how='inner')
            merged_df['Date'] = pd.to_datetime(merged_df['Date'])
            merged_df = merged_df.drop_duplicates(subset='Date', keep='first').reset_index(drop=True)

            # Prepare data for Prophet
            df_prophet = merged_df.rename(columns={'Date': 'ds', 'Stock Price': 'y', 'VIX': 'vix'})
            df_prophet['vix'] = df_prophet['vix'].fillna(method='ffill')

            # Train and forecast using Prophet
            model = Prophet(yearly_seasonality=True)
            model.add_seasonality(name='weekly', period=7, fourier_order=3)
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_regressor('vix')
            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=5, freq='D', include_history=True)
            future['vix'] = df_prophet['vix'].iloc[-1]
            forecast = model.predict(future)

            # Store results
            results[ticker] = {
                "forecast": forecast,
                "df_prophet": df_prophet,
                "model": model,
            }

            # Evaluation Metrics for each ticker
            actual_values = df_prophet['y'][-len(forecast):]
            predicted_values = forecast['yhat'][:len(actual_values)]

            mse = mean_squared_error(actual_values, predicted_values)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual_values, predicted_values)
            percent_mse = (mse / np.mean(actual_values)) * 100

            # Accumulate errors for all tickers
            all_mse.append(mse)
            all_rmse.append(rmse)
            all_r2.append(r2)
            all_percent_mse.append(percent_mse)

            # Optionally, you could print individual ticker's metrics if needed
            st.write(f"Evaluation metrics for {ticker}:")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            st.write(f"R-squared Score (R²): {r2:.2f}")
            st.write(f"Average Percentage MSE (%MSE): {percent_mse:.2f}%")

        # Overall Evaluation Metrics (across all tickers)
        avg_mse = np.mean(all_mse)
        avg_rmse = np.mean(all_rmse)
        avg_r2 = np.mean(all_r2)
        avg_percent_mse = np.mean(all_percent_mse)

        # Display overall evaluation
        st.write(f"\nModel Evaluation (Overall across all tickers):")
        st.write(f"Mean Squared Error (MSE): {avg_mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {avg_rmse:.2f}")
        st.write(f"R-squared Score (R²): {avg_r2:.2f}")
        st.write(f"Average Percentage MSE (%MSE): {avg_percent_mse:.2f}%")

        # Display if the model is a good fit or not
        if avg_rmse < 10:  # Example threshold
            st.success("The model is a good fit!")
        else:
            st.warning("The model is not a good fit. Consider adjusting parameters.")
        
        st.write("Processing complete for all tickers.")
    else:
        st.error("Failed to fetch stock or VIX data.")
