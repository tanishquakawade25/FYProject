import streamlit as st
import yfinance as yf
import pandas as pd

def stock_data():
    # Initialize the active state
    # st.session_state.active_page = "Fetch Data"
    st.title("Analysing")

    # Sidebar input for stock tickers and date range
    st.sidebar.header("Add Stocks to Analyze")
    manual_stocks = st.sidebar.text_input(
        "Enter Stock Tickers (comma-separated, e.g., RELIANCE.NS,TCS.NS):", ""
    )
    
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-07-18"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-11-10"))

    # Extract stock tickers
    stocks_to_analyze = [ticker.strip() for ticker in manual_stocks.split(",") if ticker.strip()]

    # Button to fetch data
    if st.sidebar.button("Fetch Data"):
        if start_date >= end_date:
            st.error("Start date must be before the end date!")
        elif not stocks_to_analyze:
            st.warning("Please enter at least one stock ticker.")
        else:
            # Fetch stock data
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

                # Show the merged data
                st.write("Merged Data:")
                st.dataframe(merged_data)

                # Allow downloading the merged data as CSV
                csv_data = merged_data.to_csv(index=False)
                st.download_button(
                    label="Download Merged Data",
                    data=csv_data,
                    file_name="merged_stock_data.csv",
                    mime="text/csv",
                )
            else:
                st.error("No valid data available to display.")

def app(): 
    stock_data()



# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns


# # Streamlit app title
# st.title("Stock and VIX Index Visualization")


# # Sidebar for selecting stocks manually
# st.sidebar.header("Add Stocks to Analyze")


# # Text input for adding stock tickers
# manual_stocks = st.sidebar.text_input(
#     "Enter Stock Tickers (comma-separated, e.g., RELIANCE.NS,TCS.NS):",
#     ""
# )


# # Date range input
# start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-07-18"))
# end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-11-10"))


# # Process the user input to extract stock tickers
# stocks_to_analyze = [ticker.strip() for ticker in manual_stocks.split(",") if ticker.strip()]


# # Initialize stock data storage
# if "stock_data" not in st.session_state:
#     st.session_state.stock_data = pd.DataFrame()


# # Button to fetch data
# def app(): 
#     if st.sidebar.button("Fetch Data"):
#         st.session_state["data_fetch"]=True

#         if start_date >= end_date:
#             st.error("Start date must be before the end date!")
#         elif not stocks_to_analyze:
#             st.warning("Please enter at least one stock ticker.")
#         else:
#             # Fetch historical stock data for entered tickers
#             stock_data = pd.DataFrame()
#             for ticker in stocks_to_analyze:
#                 try:
#                     data = yf.Ticker(ticker).history(start=start_date, end=end_date)
#                     if not data.empty:
#                         stock_data[ticker] = data['Close']
#                     else:
#                         st.warning(f"No data found for ticker: {ticker}")
#                 except Exception as e:
#                     st.warning(f"Could not fetch data for ticker: {ticker}. Error: {e}")


#             # Fetch VIX data
#             try:
#                 vix_data = yf.Ticker("^VIX").history(start=start_date, end=end_date)
#                 vix_data = vix_data[['Close']].rename(columns={'Close': 'VIX Index'})
#             except Exception as e:
#                 st.error(f"Could not fetch VIX data. Error: {e}")
#                 vix_data = pd.DataFrame()


#             if not stock_data.empty and not vix_data.empty:
#                 # Align VIX data with stock data by date
#                 stock_data['Date'] = stock_data.index.date
#                 vix_data['Date'] = vix_data.index.date
#                 stock_data.reset_index(drop=True, inplace=True)
#                 vix_data.reset_index(drop=True, inplace=True)


#                 # Merge the stock data with VIX data
#                 merged_data = pd.merge(stock_data, vix_data, on="Date", how="inner")


#                 # Store merged data in session state
#                 st.session_state.stock_data = merged_data


#                 # Show the stock data
#                 st.write("Stock Data Overview:")
#                 st.dataframe(merged_data.head())


#                 # Allow user to download the stock data as a CSV file
#                 csv_data = merged_data.to_csv(index=False)
#                 st.download_button(
#                     label="Download Stock Data as CSV",
#                     data=csv_data,
#                     file_name="stock_data.csv",
#                     mime="text/csv",
#                 )
#             else:
#                 st.error("No valid data available to display.")



