import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Function to fetch hourly VIX data with a period
def fetch_hourly_vix_data_with_period(ticker, start_date, end_date):
    all_data = []  # Store data chunks
    current_start = pd.Timestamp(start_date)
    final_end = pd.Timestamp(end_date)

    while current_start < final_end:
        try:
            vix = yf.Ticker(ticker)
            data = vix.history(interval="1h", period="1d", start=current_start.date(), end=(current_start + pd.Timedelta(days=1)).date())
            if not data.empty:
                all_data.append(data)
            else:
                print(f"No data found for {ticker} on {current_start.date()}")
        except Exception as e:
            print(f"Error fetching data for {ticker} on {current_start.date()}: {e}")
        
        current_start += pd.Timedelta(days=1)  # Move to the next day

    # Combine all chunks into a single DataFrame
    if all_data:
        combined_data = pd.concat(all_data)
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]  # Remove duplicates
        return combined_data
    else:
        print(f"Data for {ticker} not found for the given date range.")
        return pd.DataFrame()  # Return empty DataFrame

# Function for Random Forest Model
def run_random_forest_model(stocks_input, start_date, end_date):
    # Streamlit app title
    st.subheader("Stock Price Predictions using Random Forest Regression Model")

    # Two more options to choose from: Tabular or Graphical
    st.write("Choose Visualisation Mode : ")
    prediction_mode = st.radio(
        "",
        ("Tabular", "Graphical"),
        horizontal=True
    )

    # Process the user input to extract stock tickers
    stocks_to_analyze = [ticker.strip() for ticker in stocks_input.split(",") if ticker.strip()]

    # Validate tickers before fetching data
    invalid_tickers = []
    valid_tickers = []

    def validate_ticker(ticker):
        try:
            data = yf.Ticker(ticker).history(period="1d")
            if data.empty:
                return False, "No data found for the ticker."
            return True, "Ticker is valid."
        except Exception as e:
            return False, f"Error fetching data for ticker '{ticker}': {e}"

    for ticker in stocks_to_analyze:
        is_valid, message = validate_ticker(ticker)
        if is_valid:
            valid_tickers.append(ticker)
        else:
            invalid_tickers.append((ticker, message))

    # Display messages for invalid tickers
    if invalid_tickers:
        for ticker, message in invalid_tickers:
            st.warning(f"Invalid ticker '{ticker}': {message}")

    # Ensure there are valid tickers
    if not valid_tickers:
        st.warning("No valid tickers found. Please check your input or data availability.")
        st.stop()  # Stop the app if no valid tickers

    # Fetch VIX data
    vix_ticker = "^INDIAVIX"
    try:
        vix_data = fetch_hourly_vix_data_with_period(vix_ticker, start_date, end_date)
        vix_data = vix_data[['Close']].rename(columns={'Close': 'India VIX'})
    except ValueError as e:
        st.error(f"Error fetching VIX data: {e}")
        st.stop()

    # Fetch stock data for valid tickers
    stock_data = {}
    for ticker in valid_tickers:
        try:
            data = fetch_hourly_vix_data_with_period(ticker, start_date, end_date)
            if not data.empty:
                stock_data[ticker] = data[['Close']].rename(columns={'Close': ticker})
        except ValueError as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            st.stop()

    # Combine all data into a single DataFrame
    combined_data = vix_data
    for ticker, data in stock_data.items():
        combined_data = combined_data.join(data, how='outer')

    # Sort and process data
    combined_data.sort_index(inplace=True)
    combined_data.index = combined_data.index.tz_convert('Asia/Kolkata')  # Convert to IST
    combined_data.dropna(inplace=True)  # Drop rows with missing values

    # Feature Engineering: Lag Features
    for ticker in valid_tickers:
        combined_data[f'{ticker}_Close_Lag1'] = combined_data[ticker].shift(1)
    combined_data['VIX_Close_Lag1'] = combined_data['India VIX'].shift(1)
    combined_data.dropna(inplace=True)

    # Initialize aggregate metrics
    total_mse = 0
    total_rmse = 0
    total_r2 = 0
    total_percent_mse = 0
    total_tickers = len(valid_tickers)

    # Train Random Forest Model and make predictions
    predictions = {}

    for ticker in valid_tickers:
        X = combined_data[[f'{ticker}_Close_Lag1', 'VIX_Close_Lag1']]
        y = combined_data[ticker]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2_score_value = model.score(X_test, y_test)

        # Calculate %MSE (Percentage Mean Squared Error)
        variance_actual_values = np.var(y_test)
        percent_mse = (mse / variance_actual_values) * 100

        # Aggregate the results
        total_mse += mse
        total_rmse += rmse
        total_r2 += r2_score_value
        total_percent_mse += percent_mse

        # Predict future stock prices (Recursive Prediction)
        future_dates = pd.date_range(start=combined_data.index[-1] + pd.Timedelta(hours=1), periods=5)  # Next 5 hours
        future_predictions = []
        last_known_date = combined_data.index[-1]
        last_known_ticker_price = combined_data.iloc[-1][ticker]
        last_known_vix = combined_data.iloc[-1]['India VIX']

        for date in future_dates:
            input_features = pd.DataFrame([[last_known_ticker_price, last_known_vix]], columns=[f'{ticker}_Close_Lag1', 'VIX_Close_Lag1'])
            predicted_ticker = model.predict(input_features)[0]
            future_predictions.append(predicted_ticker)
            last_known_ticker_price = predicted_ticker

        predictions[ticker] = (future_dates, future_predictions)

    # Calculate the average of the aggregated metrics
    avg_mse = total_mse / total_tickers
    avg_rmse = total_rmse / total_tickers
    avg_r2 = total_r2 / total_tickers
    avg_percent_mse = total_percent_mse / total_tickers

    # Display results based on selected visualization mode
    if prediction_mode == "Tabular":
        # Show predictions in tabular format
        for ticker in valid_tickers:
            future_dates, future_predictions = predictions[ticker]
            future_df = pd.DataFrame({'Date': future_dates, f'Predicted_{ticker}_Close': future_predictions})

            st.write(f"\nPredicted {ticker} Close Prices for the Next 5 Days:")
            st.dataframe(future_df, use_container_width=True)

            # Allow user to download the stock data as a CSV file
            csv_data = future_df.to_csv(index=True)  # Include the index (timestamp) in the CSV
            st.download_button(
                label=f"Download Predicted {ticker} Values as CSV",
                data=csv_data,
                file_name=f"Predicted_{ticker}_Values.csv",
                mime="text/csv",
                icon="⬇️",
            )

    elif prediction_mode == "Graphical":
        # Show predictions as a graph
        for ticker in valid_tickers:
            future_dates, future_predictions = predictions[ticker]
            future_df = pd.DataFrame({'Date': future_dates, f'Predicted_{ticker}_Close': future_predictions})

            # Plot historical and predicted data
            plt.figure(figsize=(20, 10))
            plt.plot(combined_data.index, combined_data[ticker], label=f"Historical {ticker} Close", color="blue")
            plt.plot(future_df['Date'], future_df[f'Predicted_{ticker}_Close'], label=f"Predicted Future {ticker}_Close", color="green", linestyle="--", marker="o")
            plt.title(f"{ticker} Close Price: Historical and Forecast")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid()

            # Save the plot to an in-memory file
            image_path = f"predicted_{ticker}_close_price_plot.png"
            plt.savefig(image_path, format="png")

            # Display the plot
            st.pyplot(plt)

            # Allow user to download the plot as an image
            with open(image_path, "rb") as img_file:
                st.download_button(
                    label=f"Download {ticker} Plot as Image",
                    data=img_file,
                    file_name=image_path,
                    mime="image/png",
                    icon="⬇️",
                )

    # Display the overall evaluation metrics
    st.write(f"\nModel Evaluation (Overall):")
    st.write(f"Mean Squared Error (MSE): {avg_mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {avg_rmse:.2f}")
    st.write(f"R-squared Score (R²): {avg_r2:.2f}")
    st.write(f"Average Percentage MSE (%MSE): {avg_percent_mse:.2f}%")









# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import streamlit as st
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# import yfinance as yf

# # Function to fetch hourly VIX data with a period
# def fetch_hourly_vix_data_with_period(ticker, start_date, end_date):
#     all_data = []  # Store data chunks
#     current_start = pd.Timestamp(start_date)
#     final_end = pd.Timestamp(end_date)

#     while current_start < final_end:
#         try:
#             vix = yf.Ticker(ticker)
#             data = vix.history(interval="1h", period="1d", start=current_start.date(), end=(current_start + pd.Timedelta(days=1)).date())
#             if not data.empty:
#                 all_data.append(data)
#             else:
#                 print(f"No data found for {ticker} on {current_start.date()}")
#         except Exception as e:
#             print(f"Error fetching data for {ticker} on {current_start.date()}: {e}")
        
#         current_start += pd.Timedelta(days=1)  # Move to the next day

#     # Combine all chunks into a single DataFrame
#     if all_data:
#         combined_data = pd.concat(all_data)
#         combined_data = combined_data[~combined_data.index.duplicated(keep='first')]  # Remove duplicates
#         return combined_data
#     else:
#         print(f"Data for {ticker} not found for the given date range.")
#         return pd.DataFrame()  # Return empty DataFrame

# # Function for Random Forest Model
# def run_random_forest_model():
#     # Streamlit app title
#     st.subheader("Stock Price Predictions using Random Forest Regression Model")

#     # Two more options to choose from: Tabular or Graphical
#     prediction_mode = st.radio(
#         "Choose Visualisation Mode : ",
#         ("Tabular", "Graphical"),
#         horizontal=True
#     )

#     # Sidebar for selecting stocks manually
#     st.sidebar.header("Add Stocks to Analyze")
#     manual_stocks = st.sidebar.text_input(
#         "Enter Stock Tickers (comma-separated, e.g., RELIANCE.NS,TCS.NS):",
#         ""
#     )

#     # Date range input
#     start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-11-25"))
#     end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-10"))

#     # Process the user input to extract stock tickers
#     stocks_to_analyze = [ticker.strip() for ticker in manual_stocks.split(",") if ticker.strip()]

#     # Validate tickers before fetching data
#     invalid_tickers = []
#     valid_tickers = []

#     def validate_ticker(ticker):
#         try:
#             data = yf.Ticker(ticker).history(period="1d")
#             if data.empty:
#                 return False, "No data found for the ticker."
#             return True, "Ticker is valid."
#         except Exception as e:
#             return False, f"Error fetching data for ticker '{ticker}': {e}"

#     for ticker in stocks_to_analyze:
#         is_valid, message = validate_ticker(ticker)
#         if is_valid:
#             valid_tickers.append(ticker)
#         else:
#             invalid_tickers.append((ticker, message))

#     # Display messages for invalid tickers
#     if invalid_tickers:
#         for ticker, message in invalid_tickers:
#             st.warning(f"Invalid ticker '{ticker}': {message}")

#     # Ensure there are valid tickers
#     if not valid_tickers:
#         st.warning("No valid tickers found. Please check your input or data availability.")
#         st.stop()  # Stop the app if no valid tickers

#     # Fetch VIX data
#     vix_ticker = "^INDIAVIX"
#     try:
#         vix_data = fetch_hourly_vix_data_with_period(vix_ticker, start_date, end_date)
#         vix_data = vix_data[['Close']].rename(columns={'Close': 'India VIX'})
#     except ValueError as e:
#         st.error(f"Error fetching VIX data: {e}")
#         st.stop()

#     # Fetch stock data for valid tickers
#     stock_data = {}
#     for ticker in valid_tickers:
#         try:
#             data = fetch_hourly_vix_data_with_period(ticker, start_date, end_date)
#             if not data.empty:
#                 stock_data[ticker] = data[['Close']].rename(columns={'Close': ticker})
#         except ValueError as e:
#             st.error(f"Error fetching data for {ticker}: {e}")
#             st.stop()

#     # Combine all data into a single DataFrame
#     combined_data = vix_data
#     for ticker, data in stock_data.items():
#         combined_data = combined_data.join(data, how='outer')

#     # Sort and process data
#     combined_data.sort_index(inplace=True)
#     combined_data.index = combined_data.index.tz_convert('Asia/Kolkata')  # Convert to IST
#     combined_data.dropna(inplace=True)  # Drop rows with missing values

#     # Feature Engineering: Lag Features
#     for ticker in valid_tickers:
#         combined_data[f'{ticker}_Close_Lag1'] = combined_data[ticker].shift(1)
#     combined_data['VIX_Close_Lag1'] = combined_data['India VIX'].shift(1)
#     combined_data.dropna(inplace=True)

#     # Initialize aggregate metrics
#     total_mse = 0
#     total_rmse = 0
#     total_r2 = 0
#     total_percent_mse = 0
#     total_tickers = len(valid_tickers)

#     # Train Random Forest Model and make predictions
#     predictions = {}

#     for ticker in valid_tickers:
#         X = combined_data[[f'{ticker}_Close_Lag1', 'VIX_Close_Lag1']]
#         y = combined_data[ticker]

#         # Train-test split
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#         model.fit(X_train, y_train)

#         # Make predictions
#         y_pred = model.predict(X_test)
        
#         # Evaluate the model
#         mse = mean_squared_error(y_test, y_pred)
#         rmse = np.sqrt(mse)
#         r2_score_value = model.score(X_test, y_test)

#         # Calculate %MSE (Percentage Mean Squared Error)
#         variance_actual_values = np.var(y_test)
#         percent_mse = (mse / variance_actual_values) * 100

#         # Aggregate the results
#         total_mse += mse
#         total_rmse += rmse
#         total_r2 += r2_score_value
#         total_percent_mse += percent_mse

#         # Predict future stock prices (Recursive Prediction)
#         future_dates = pd.date_range(start=combined_data.index[-1] + pd.Timedelta(hours=1), periods=5)  # Next 5 hours
#         future_predictions = []
#         last_known_date = combined_data.index[-1]
#         last_known_ticker_price = combined_data.iloc[-1][ticker]
#         last_known_vix = combined_data.iloc[-1]['India VIX']

#         for date in future_dates:
#             input_features = pd.DataFrame([[last_known_ticker_price, last_known_vix]], columns=[f'{ticker}_Close_Lag1', 'VIX_Close_Lag1'])
#             predicted_ticker = model.predict(input_features)[0]
#             future_predictions.append(predicted_ticker)
#             last_known_ticker_price = predicted_ticker

#         predictions[ticker] = (future_dates, future_predictions)

#     # Calculate the average of the aggregated metrics
#     avg_mse = total_mse / total_tickers
#     avg_rmse = total_rmse / total_tickers
#     avg_r2 = total_r2 / total_tickers
#     avg_percent_mse = total_percent_mse / total_tickers

#     # Display results based on selected visualization mode
#     if prediction_mode == "Tabular":
#         # Show predictions in tabular format
#         for ticker in valid_tickers:
#             future_dates, future_predictions = predictions[ticker]
#             future_df = pd.DataFrame({'Date': future_dates, f'Predicted_{ticker}_Close': future_predictions})

#             st.write(f"\nPredicted {ticker} Close Prices for the Next 5 Days:")
#             st.write(future_df)

#             # Allow user to download the stock data as a CSV file
#             csv_data = future_df.to_csv(index=True)  # Include the index (timestamp) in the CSV
#             st.download_button(
#                 label=f"Download Predicted {ticker} Values as CSV",
#                 data=csv_data,
#                 file_name=f"Predicted_{ticker}_Values.csv",
#                 mime="text/csv",
#                 icon="⬇️",
#             )

#     elif prediction_mode == "Graphical":
#         # Show predictions as a graph
#         for ticker in valid_tickers:
#             future_dates, future_predictions = predictions[ticker]
#             future_df = pd.DataFrame({'Date': future_dates, f'Predicted_{ticker}_Close': future_predictions})

#             # Plot historical and predicted data
#             plt.figure(figsize=(20, 10))
#             plt.plot(combined_data.index, combined_data[ticker], label=f"Historical {ticker} Close", color="blue")
#             plt.plot(future_df['Date'], future_df[f'Predicted_{ticker}_Close'], label=f"Predicted Future {ticker}_Close", color="green", linestyle="--", marker="o")
#             plt.title(f"{ticker} Close Price: Historical and Forecast")
#             plt.xlabel("Date")
#             plt.ylabel("Price")
#             plt.legend()
#             plt.grid()

#             # Save the plot to an in-memory file
#             image_path = f"predicted_{ticker}_close_price_plot.png"
#             plt.savefig(image_path, format="png")

#             # Display the plot
#             st.pyplot(plt)

#             # Allow user to download the plot as an image
#             with open(image_path, "rb") as img_file:
#                 st.download_button(
#                     label=f"Download {ticker} Plot as Image",
#                     data=img_file,
#                     file_name=image_path,
#                     mime="image/png",
#                     icon="⬇️"
#                 )

#             # Close the figure to avoid memory issues with multiple plots
#             plt.close()


#     # Display the overall evaluation metrics
#     st.write(f"\nModel Evaluation (Overall):")
#     st.write(f"Mean Squared Error (MSE): {avg_mse:.2f}")
#     st.write(f"Root Mean Squared Error (RMSE): {avg_rmse:.2f}")
#     st.write(f"R-squared Score (R²): {avg_r2:.2f}")
#     st.write(f"Average Percentage MSE (%MSE): {avg_percent_mse:.2f}%")
    
# # For standalone testing
# if __name__ == "__main__":
#     run_random_forest_model()
