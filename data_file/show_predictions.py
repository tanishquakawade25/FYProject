import streamlit as st
import pandas as pd
import sys
sys.path.append('C:/Users/hp/Desktop/FYProject')
from data_file import prophet_model

# Main Streamlit app
def app():
    print("Show predictions is running")
    
    # Collect user input for tickers and date range
    manual_stocks = st.sidebar.text_input(
        "Enter Stock Tickers (e.g., RELIANCE.NS,TCS.NS):",
        ""
    )
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-11-25"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-10"))
    
    
    print("Prophet model is running")
    
    st.write("🛈 Prophet is a forecasting model designed to predict time series data with strong seasonal patterns and holidays. It uses additive models to capture trends, seasonality, and holidays, providing accurate predictions for future stock prices.")
    
    # Call the prophet model with user inputs
    prophet_model.run_prophet_model(manual_stocks, start_date, end_date)