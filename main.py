import streamlit as st
from apps import sign_up, login, home

st.set_page_config(
    page_title="Stock Market App",
    page_icon="chart_with_upwards_trend",
)

def main():
    
    # Initialize session state variables
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = "login"  # Default to login page
    
    # Navigation logic
    if st.session_state.logged_in:
        home.app()  # Redirect to home page if logged in
    else:
        if st.session_state.current_page == "login":
            login.app()
        if st.session_state.current_page == "sign_up":
            sign_up.app()

if __name__ == "__main__":
    main()



