# NTPC.NS, ONGC.NS, JSWSTEEL.NS, TATAPOWER.NS


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title
st.title("Stock and VIX Index Visualization")

# Sidebar for selecting stocks manually
st.sidebar.header("Add Stocks to Analyze")

# Text input for adding stock tickers
manual_stocks = st.sidebar.text_input(
    "Enter Stock Tickers (comma-separated, e.g., RELIANCE.NS,TCS.NS):",
    ""
)

# Date range input
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-07-18"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-11-10"))

# Process the user input to extract stock tickers
stocks_to_analyze = [ticker.strip() for ticker in manual_stocks.split(",") if ticker.strip()]

# Initialize stock data storage
if "stock_data" not in st.session_state:
    st.session_state.stock_data = pd.DataFrame()

# Button to fetch data
if st.sidebar.button("Fetch Data"):
    if start_date >= end_date:
        st.error("Start date must be before the end date!")
    elif not stocks_to_analyze:
        st.warning("Please enter at least one stock ticker.")
    else:
        # Fetch historical stock data for entered tickers
        stock_data = pd.DataFrame()
        for ticker in stocks_to_analyze:
            try:
                data = yf.Ticker(ticker).history(start=start_date, end=end_date)
                if not data.empty:
                    stock_data[ticker] = data['Close']
                else:
                    st.warning(f"No data found for ticker: {ticker}")
            except Exception as e:
                st.warning(f"Could not fetch data for ticker: {ticker}. Error: {e}")

        # Fetch VIX data
        try:
            vix_data = yf.Ticker("^VIX").history(start=start_date, end=end_date)
            vix_data = vix_data[['Close']].rename(columns={'Close': 'VIX Index'})
        except Exception as e:
            st.error(f"Could not fetch VIX data. Error: {e}")
            vix_data = pd.DataFrame()

        if not stock_data.empty and not vix_data.empty:
            # Align VIX data with stock data by date
            stock_data['Date'] = stock_data.index.date
            vix_data['Date'] = vix_data.index.date
            stock_data.reset_index(drop=True, inplace=True)
            vix_data.reset_index(drop=True, inplace=True)

            # Merge the stock data with VIX data
            merged_data = pd.merge(stock_data, vix_data, on="Date", how="inner")

            # Store merged data in session state
            st.session_state.stock_data = merged_data

            # Show the stock data
            st.write("Stock Data Overview:")
            st.dataframe(merged_data.head())

            # Allow user to download the stock data as a CSV file
            csv_data = merged_data.to_csv(index=False)
            st.download_button(
                label="Download Stock Data as CSV",
                data=csv_data,
                file_name="stock_data.csv",
                mime="text/csv",
            )
        else:
            st.error("No valid data available to display.")

# Button to visualize stock prices and VIX Index
if st.sidebar.button("Visualize Stock Prices and VIX"):
    if st.session_state.stock_data.empty:
        st.warning("No data available. Please fetch the data first.")
    else:
        try:
            # Retrieve the merged data from session state
            merged_data = st.session_state.stock_data

            # Create figure and axis
            fig, ax1 = plt.subplots(figsize=(12, 8))

            # Plot stock prices (Y1 axis)
            for stock in stocks_to_analyze:
                if stock in merged_data.columns:
                    ax1.plot(merged_data['Date'], merged_data[stock], label=stock)

            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Stock Prices', fontsize=12)
            ax1.set_title('Stock Prices and VIX Index over Time', fontsize=16)
            ax1.tick_params(axis='x', rotation=45)

            # Create second y-axis for VIX Index
            ax2 = ax1.twinx()
            ax2.plot(merged_data['Date'], merged_data['VIX Index'], color='black', label='VIX Index', linestyle='--')

            ax2.set_ylabel('VIX Index', fontsize=12)

            # Add legends
            ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Stock Prices')
            ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.8), title='VIX Index')

            plt.tight_layout()  # Adjust layout for better fitting
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error in plotting data: {e}")

# Button to calculate Spearman correlation between selected stocks
if st.sidebar.button("Find Correlation"):
    if st.session_state.stock_data.empty:
        st.warning("No data available. Please fetch the data first.")
    else:
        try:
            # Retrieve the merged data from session state
            merged_data = st.session_state.stock_data

            # Ensure VIX Index is included in the correlation calculation
            columns_to_analyze = stocks_to_analyze + ["VIX Index"]
            filtered_data = merged_data[columns_to_analyze]

            if filtered_data.shape[1] < 2:
                st.warning("Please select at least two stocks to calculate correlation.")
            else:
                # Calculate Spearman Correlation Matrix manually
                correlation_matrix = pd.DataFrame(
                    index=filtered_data.columns,
                    columns=filtered_data.columns,
                )

                def spearman_manual(x, y):
                    # Step 1: Rank the data
                    x_rank = pd.Series(x).rank(method='average')
                    y_rank = pd.Series(y).rank(method='average')

                    # Step 2: Compute the difference between ranks
                    d = x_rank - y_rank
                    d_squared = d ** 2

                    # Step 3: Apply the formula
                    n = len(x)
                    numerator = 6 * np.sum(d_squared)
                    denominator = n * (n ** 2 - 1)
                    return 1 - (numerator / denominator)

                # Fill the matrix manually
                for col1 in filtered_data.columns:
                    for col2 in filtered_data.columns:
                        x = filtered_data[col1].dropna()
                        y = filtered_data[col2].dropna()
                        common_index = x.index.intersection(y.index)  # Align data by index
                        if len(common_index) > 1:  # Ensure sufficient data
                            correlation_matrix.loc[col1, col2] = spearman_manual(
                                x[common_index], y[common_index]
                            )
                        else:
                            correlation_matrix.loc[col1, col2] = np.nan  # Not enough data

                correlation_matrix = correlation_matrix.astype(float)

                # Display the correlation matrix
                st.write("Spearman Correlation Matrix (Selected Stocks and VIX Index):")
                st.dataframe(correlation_matrix)

                # Visualize the correlation matrix using a heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
                ax.set_title("Spearman Correlation Matrix of Selected Stocks and VIX Index")
                st.pyplot(fig)

                # Allow user to download the correlation matrix as a CSV
                csv_corr = correlation_matrix.to_csv()
                st.download_button(
                    label="Download Correlation Matrix as CSV",
                    data=csv_corr,
                    file_name="spearman_correlation_matrix.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error computing correlation: {e}")

# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import matplotlib.pyplot as plt

# # Streamlit app title
# st.title("Stock and VIX Index Visualization")

# # Sidebar for selecting sectors and stocks
# st.sidebar.header("Filters")

# # Correct list of stock tickers for each sector
# sectors = {
#     "IT": ["TCS.NS", "INFY.NS", "TECHM.NS", "BSOFT.NS"],
#     "Energy": ["BPCL.NS", "ONGC.NS", "JSWSTEEL.NS", "TATAPOWER.NS", "RELIANCE.NS"],
#     # Add more sectors here as needed
# }

# # Dropdown for selecting sectors
# selected_sectors = st.sidebar.multiselect("Select Sectors", options=sectors.keys())

# # Dynamically populate stocks based on selected sectors
# selected_stocks = []
# if selected_sectors:
#     for sector in selected_sectors:
#         selected_stocks.extend(sectors[sector])

# # Dropdown for selecting individual stocks
# stocks_to_analyze = st.sidebar.multiselect("Select Stocks", options=selected_stocks)

# # Date range input
# start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-07-18"))
# end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-11-10"))

# # Initialize stock data storage
# if "stock_data" not in st.session_state:
#     st.session_state.stock_data = pd.DataFrame()

# # Button to fetch data
# if st.sidebar.button("Fetch Data"):
#     if start_date >= end_date:
#         st.error("Start date must be before the end date!")
#     elif not stocks_to_analyze:
#         st.warning("Please select at least one stock.")
#     else:
#         # Fetch historical stock data for selected stocks
#         stock_data = pd.DataFrame()
#         for ticker in stocks_to_analyze:
#             try:
#                 data = yf.Ticker(ticker).history(start=start_date, end=end_date)
#                 if not data.empty:
#                     stock_data[ticker] = data['Close']
#                 else:
#                     st.warning(f"No data found for ticker: {ticker}")
#             except Exception as e:
#                 st.warning(f"Could not fetch data for ticker: {ticker}. Error: {e}")

#         # Fetch VIX data
#         try:
#             vix_data = yf.Ticker("^VIX").history(start=start_date, end=end_date)
#             vix_data = vix_data[['Close']].rename(columns={'Close': 'VIX Index'})
#         except Exception as e:
#             st.error(f"Could not fetch VIX data. Error: {e}")
#             vix_data = pd.DataFrame()

#         if not stock_data.empty and not vix_data.empty:
#             # Align VIX data with stock data by date
#             stock_data['Date'] = stock_data.index.date
#             vix_data['Date'] = vix_data.index.date
#             stock_data.reset_index(drop=True, inplace=True)
#             vix_data.reset_index(drop=True, inplace=True)

#             # Merge the stock data with VIX data
#             merged_data = pd.merge(stock_data, vix_data, on="Date", how="inner")

#             # Store merged data in session state
#             st.session_state.stock_data = merged_data

#             # Show the stock data
#             st.write("Stock Data Overview:")
#             st.dataframe(merged_data.head())

#             # Allow user to download the stock data as a CSV file
#             csv_data = merged_data.to_csv()
#             st.download_button(
#                 label="Download Stock Data as CSV",
#                 data=csv_data,
#                 file_name="stock_data.csv",
#                 mime="text/csv",
#             )
#         else:
#             st.error("No valid data available to display.")

# # Button to visualize stock prices and VIX Index
# if st.sidebar.button("Visualize Stock Prices and VIX"):
#     if st.session_state.stock_data.empty:
#         st.warning("No data available. Please fetch the data first.")
#     else:
#         try:
#             # Retrieve the merged data from session state
#             merged_data = st.session_state.stock_data

#             # Create figure and axis
#             fig, ax1 = plt.subplots(figsize=(12, 8))

#             # Plot stock prices (Y1 axis)
#             for stock in stocks_to_analyze:
#                 if stock in merged_data.columns:
#                     ax1.plot(merged_data['Date'], merged_data[stock], label=stock)

#             ax1.set_xlabel('Date', fontsize=12)
#             ax1.set_ylabel('Stock Prices', fontsize=12)
#             ax1.set_title('Stock Prices and VIX Index over Time', fontsize=16)
#             ax1.tick_params(axis='x', rotation=45)

#             # Create second y-axis for VIX Index
#             ax2 = ax1.twinx()
#             ax2.plot(merged_data['Date'], merged_data['VIX Index'], color='black', label='VIX Index', linestyle='--')

#             ax2.set_ylabel('VIX Index', fontsize=12)

#             # Add legends
#             ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.9), title='VIX Index')
#             ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Stock Prices')
            
#             plt.tight_layout()  # Adjust layout for better fitting
#             st.pyplot(fig)

#         except Exception as e:
#             st.error(f"Error in plotting data: {e}")


