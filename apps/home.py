import streamlit as st
from data_file import data_fetcher, find_correlation, indiavix_data_fetcher, show_predictions

def app():
    if st.session_state.get("logged_in", False):
        st.title("Welcome to the Stock Market App")

        # Sidebar navigation
        st.sidebar.subheader("Navigation")
                
        if st.sidebar.button("Home"):
            st.session_state.active_page = "home"
        
        if st.sidebar.button("Stock Analysis"):
            # Set session state for stock analysis
            st.session_state.active_page = "stock_analysis"
            st.write("Explore stock and VIX data in interactive tables or insightful graphs, and easily download the data or chart for further analysis.")


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
        
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()  # Log out and reload to the login page
                    
    else:
        st.warning("You must log in to access this page.")
        st.rerun()    
            
    

