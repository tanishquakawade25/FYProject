import streamlit as st
import pandas as pd
import sys
sys.path.append('C:/Users/hp/Desktop/FYProject')
from data_file import random_forest_model
from data_file import prophet_model

# Main Streamlit app
def app():
    print("Show predictions is running")
    
    # Collect user input for tickers and date range
    manual_stocks = st.sidebar.text_input(
        "Enter Stock Tickers (comma-separated, e.g., RELIANCE.NS,TCS.NS):",
        ""
    )
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-11-25"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-10"))
    
    st.subheader("Choose a Model:")
    view_option = st.radio(
        "  ",
        ("Random Forest Model", "Prophet Model"),
        horizontal=True  # Makes the radio buttons appear side by side
    )

    if view_option == "Random Forest Model":
        print("Random Forest Model is running")

        st.write("Random Forest is an ensemble machine learning algorithm used for predictions for more accurate and robust results.Random Forest predicts stock prices by combining multiple decision trees to learn complex patterns and improve accuracy.")

        # Call the random forest model with user inputs
        random_forest_model.run_random_forest_model(manual_stocks, start_date, end_date)

    elif view_option == "Prophet Model":
        print("Prophet model is running")
        
        st.write("Prophet is a forecasting model designed to predict time series data with strong seasonal patterns and holidays. It uses additive models to capture trends, seasonality, and holidays, providing accurate predictions for future stock prices.")
        
        # Call the prophet model with user inputs
        prophet_model.run_prophet_model(manual_stocks, start_date, end_date)














# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import streamlit as st
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# import yfinance as yf

# import sys
# sys.path.append('C:/Users/hp/Desktop/FYProject')
# from data_file import prophet_model
# from data_file import random_forest_model

# # Main Streamlit app
# def app():
#     print("Show predictions is running")
#     view_option = st.radio(
#         "Choose a Model:",
#         ("Random Forest Model", "Prophet Model"),
#         horizontal=True  # Makes the radio buttons appear side by side
#     )

#     if view_option == "Random Forest Model":
#         print("prophet model is running")
#         # random_forest_model.run_random_forest_model()

#     elif view_option == "Prophet Model":
#         print("prophet model is running")
#         # prophet_model.run_prophet_model()
        


# # For standalone testing
# if __name__ == "__main__":
#     app()        