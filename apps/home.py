import sys
sys.path.append('C:/Users/hp/Desktop/FYProject')
from data_file import indiavix_data_fetcher, data_fetcher, find_correlation, show_predictions
import streamlit as st

# import importlib
# import data_file.data_fetcher
# importlib.reload(data_file.data_fetcher)
# print("data_fetcher loaded successfully!")


def app():
    if st.session_state.get("logged_in", False):
        st.title("WelCome")

        # Sidebar navigation
        st.sidebar.subheader("Navigation")
        
        if st.sidebar.button("Home"):
            st.session_state.active_page = "home"
        
        if st.sidebar.button("Stock Analysis"):
            # Set session state for stock analysis
            st.session_state.active_page = "stock_analysis"

        if st.sidebar.button("Find correlation"):
            # Set session state for stock analysis
            st.session_state.active_page = "find_correlation"
        
        if st.sidebar.button("Show Predictions"):
            # Set session state for stock analysis
            st.session_state.active_page = "show_predictions"
        
        
        # Default home page content
        if st.session_state.get("active_page", "home") == "home":
            st.subheader("Current India VIX Data")
            indiavix_data_fetcher.display_vix_data()  # Display VIX data
            
        # if st.session_state.active_page == "home":
        #     st.subheader("Current India VIX Data")
        #     indiavix_data_fetcher.display_vix_data()  # Display VIX data
                
        elif st.session_state.active_page == "stock_analysis":
            # Call the Stock Analysis page
            data_fetcher.app()  

        elif st.session_state.active_page == "find_correlation":
            # Call the Stock Analysis page
            find_correlation.app()   
            
        elif st.session_state.active_page == "show_predictions":    
            show_predictions.app()
            
            
            
            
    else:
        st.warning("Please login first!")


















