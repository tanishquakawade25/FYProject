import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas_market_calendars as mcal
from datetime import timedelta

def app():
    """
    Main function that performs the correlation analysis and visualization.
    This function should be called from home.py.
    """
    import streamlit as st

    # Sidebar input for stock tickers and date range
    st.sidebar.header("Add Stocks to Analyze")
    manual_stocks = st.sidebar.text_input(
        "Enter Stock Tickers (comma-separated, e.g., RELIANCE.NS,TCS.NS):", ""
    )

    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-07-18"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-11-10"))

    # Extract stock tickers
    stocks_to_analyze = [ticker.strip() for ticker in manual_stocks.split(",") if ticker.strip()]

    # Step 1: Initializing NSE Calendar
    nse = mcal.get_calendar('NSE')

    # Step 2: Retrieving market days (excluding weekends and holidays)
    market_days = nse.valid_days(start_date, end_date)

    # Step 4: Generate contiguous date ranges
    date_ranges = []
    current_range = [market_days[0]]

    for i in range(1, len(market_days)):
        if (market_days[i] - market_days[i - 1]).days == 1:
            current_range.append(market_days[i])
        else:
            date_ranges.append(current_range)
            current_range = [market_days[i]]

    if current_range:
        date_ranges.append(current_range)

    # Create an empty DataFrame to store stock data
    stock_data = pd.DataFrame()

    for ticker in stocks_to_analyze:
        st.write(f"\nFetching data for stock: {ticker}")
        try:
            ticker_data = pd.DataFrame()

            # Fetch data for each date range
            for date_range in date_ranges:
                range_start = date_range[0]
                range_end = date_range[-1] + timedelta(days=1)

                data = yf.Ticker(ticker).history(start=range_start, end=range_end, interval="1d")

                if not data.empty:
                    ticker_data = pd.concat([ticker_data, data[['Close']].rename(columns={'Close': ticker})])
                else:
                    st.write(f"No data for range {range_start.date()} to {range_end.date()} for {ticker}.")
            
            if not ticker_data.empty:
                # Align data with the global stock data
                stock_data = pd.merge(stock_data, ticker_data, how="outer", left_index=True, right_index=True)
            else:
                st.write(f"No data available for stock: {ticker}")

        except Exception as e:
            st.write(f"Error fetching data for {ticker}: {e}")

    # Fetch India VIX data
    india_vix_data = pd.DataFrame()

    for i, date_range in enumerate(date_ranges):
        for day in date_range:
            try:
                vix_data = yf.Ticker("^INDIAVIX").history(
                    start=day,
                    end=day + timedelta(days=1),
                    interval="1h", period="1d"
                )

                if not vix_data.empty:
                    vix_data = vix_data[['Close']].rename(columns={'Close': 'India VIX Index'})
                    vix_data['Date'] = vix_data.index
                    india_vix_data = pd.concat([india_vix_data, vix_data])
                else:
                    st.write(f"Warning: No VIX data available for {day.date()}.")
            except Exception as e:
                st.write(f"Error fetching VIX data for {day.date()}: {e}")

    if not india_vix_data.empty:
        india_vix_data.reset_index(drop=True, inplace=True)
        st.write("\nIndia VIX data fetched successfully!")
        st.write(f"Total VIX records: {len(india_vix_data)}")
    else:
        st.write("No India VIX data available.")

    # Merge and clean the data
    if not india_vix_data.empty and not stock_data.empty:
        # Reset index for both dataframes to ensure no ambiguity
        stock_data = stock_data.reset_index()
        india_vix_data = india_vix_data.reset_index()

        # Ensure the 'Date' column is properly formatted
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
        india_vix_data['Date'] = pd.to_datetime(india_vix_data['Date']).dt.date

        try:
            # Merging stock and VIX data
            merged_data = pd.merge(stock_data, india_vix_data, on="Date", how="outer")

            # Drop rows with NaN values
            merged_data_clean = merged_data.dropna()
            
            # Remove the index column that was created after resetting the index
            merged_data_clean = merged_data_clean.drop(columns=['index'])
            
            st.write("\nCleaned Data Preview:")
            st.write(merged_data_clean.head())

        except Exception as e:
            st.write(f"Error merging data: {e}")

        # Define the distance correlation function
        def distance_correlation(X, Y):
            """
            Calculate the distance correlation between two variables X and Y.
            """
            def distance_matrix(A):
                """Compute the pairwise Euclidean distance matrix.""" 
                return np.abs(A[:, None] - A[None, :])

            def double_centering(D):
                """Apply double centering to the distance matrix.""" 
                n = D.shape[0]
                row_mean = np.mean(D, axis=1, keepdims=True)
                col_mean = np.mean(D, axis=0, keepdims=True)
                total_mean = np.mean(D)
                return D - row_mean - col_mean + total_mean

            # Compute distance matrices
            X = np.atleast_1d(X)
            Y = np.atleast_1d(Y)
            n = X.shape[0]
            if n != Y.shape[0]:
                raise ValueError("Input vectors must have the same length.")
            if n < 2:
                raise ValueError("Length of input vectors must be at least 2.")

            dist_X = double_centering(distance_matrix(X))
            dist_Y = double_centering(distance_matrix(Y))

            # Compute distance covariance
            dCovXY = np.sqrt(np.sum(dist_X * dist_Y) / (n * n))
            dVarX = np.sqrt(np.sum(dist_X * dist_X) / (n * n))
            dVarY = np.sqrt(np.sum(dist_Y * dist_Y) / (n * n))

            # Compute distance correlation
            if dVarX * dVarY == 0:
                return 0
            return dCovXY / np.sqrt(dVarX * dVarY)

        # Calculate pairwise distance correlation for all columns in merged_data_clean
        columns = merged_data_clean.columns.drop("Date")  # Exclude the 'Date' column
        correlation_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)

        for col1 in columns:
            for col2 in columns:
                if merged_data_clean[col1].dropna().shape[0] >= 2 and merged_data_clean[col2].dropna().shape[0] >= 2:
                    correlation_matrix.loc[col1, col2] = distance_correlation(
                        merged_data_clean[col1].dropna().values, merged_data_clean[col2].dropna().values
                    )
                else:
                    correlation_matrix.loc[col1, col2] = np.nan

        # Create a radio button to toggle between the views
        view_option = st.radio(
            "Choose a view:",
            ("Correlation Matrix", "Heatmap", "Trend Visualization"),
            horizontal=True  # Makes the radio buttons appear side by side
        )

        # Logic to display based on the selected option
        if view_option == "Correlation Matrix":
            st.write("\nDistance Correlation Matrix:")
            st.write(correlation_matrix)
            
            # Allow user to download the stock data as a CSV file
            csv_data = correlation_matrix.to_csv(index=True)  # Include the index (timestamp) in the CSV
            st.download_button(
                label="Download Correlation Matrix as CSV",
                data=csv_data,
                file_name="Correlation Matrix.csv",
                mime="text/csv",
                icon="📄",
            )

        elif view_option == "Heatmap":
            # Clean the correlation matrix for plotting
            correlation_matrix_clean = correlation_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')

            if correlation_matrix_clean.empty:
                st.write("No valid correlation data available to display.")
            else:
                fig, ax = plt.subplots(figsize=(5, 5))  # Adjusted figure size for better display
                sns.heatmap(correlation_matrix_clean, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
                st.pyplot(fig)
                
                # Save the heatmap as an image (ensure the directory exists)
                img_path = "heatmap.png"  # You can specify an absolute path here if needed

                # Save the figure
                fig.savefig(img_path)
                
                # Allow users to download the heatmap image
                with open("heatmap.png", "rb") as img_file:
                    btn = st.download_button(
                        label="Download Heatmap Image",
                        data=img_file,
                        file_name="heatmap.png",
                        mime="image/png",
                        icon="📥",
                    )

        elif view_option == "Trend Visualization":
            st.write("### Trend of All Stocks and India VIX Index")

            # Plot all stock tickers on primary y-axis and India VIX Index on secondary y-axis
            trend_data = merged_data_clean.set_index("Date")  # Set 'Date' as index for better plotting

            if trend_data.empty:
                st.write("No data available for trend visualization.")
            else:
                import matplotlib.dates as mdates

                # Separate India VIX data and stock data
                vix_data = trend_data["India VIX Index"]
                stock_data = trend_data.drop(columns=["India VIX Index"])

                # Create the figure and axes
                fig, ax1 = plt.subplots(figsize=(14, 7))  # Larger figure for better clarity

                # Plot stock prices on primary y-axis
                for column in stock_data.columns:
                    ax1.plot(stock_data.index, stock_data[column], label=column)

                # Customize primary y-axis
                ax1.set_title("Stock Trends and India VIX Index", fontsize=16, fontweight='bold')
                ax1.set_xlabel("Date", fontsize=14)
                ax1.set_ylabel("Stock Prices", fontsize=14)
                ax1.tick_params(axis='y', labelsize=12)
                ax1.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
                ax1.legend(loc="upper left", fontsize=10)

                # Customize x-axis with a 15-day interval
                ax1.xaxis.set_major_locator(mdates.DayLocator(interval=15))  # Tick every 15 days
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # Format as 'YYYY-MM-DD'
                plt.xticks(rotation=45)

                # Create secondary y-axis for India VIX Index
                ax2 = ax1.twinx()
                ax2.plot(vix_data.index, vix_data, label="India VIX Index", color="red", linewidth=2)
                ax2.set_ylabel("India VIX Index", fontsize=14, color="red")
                ax2.tick_params(axis='y', labelcolor="red", labelsize=12)

                # Add legend for the secondary axis
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=10)

                # Display the plot
                st.pyplot(fig)
                
                # Save the heatmap as an image (ensure the directory exists)
                img_path = "stock_vix_plot.png"  # You can specify an absolute path here if needed

                # Save the figure
                fig.savefig(img_path)
                
                # Allow users to download the heatmap image
                with open("stock_vix_plot.png", "rb") as img_file:
                    btn = st.download_button(
                        label="Download Stock VIX Plot Image",
                        data=img_file,
                        file_name="heatmap.png",
                        mime="image/png",
                        icon="📥",
                    )

    else:
        st.write("No valid data available for correlation calculation.")








