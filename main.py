
import streamlit as st
from apps import login, home
from data_file import indiavix_data_fetcher

# Main function to route the app
def main():
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.page = "login"

    # Show appropriate page
    if st.session_state.page == "home":
        home.app()
    else:
        login.app()

if __name__ == "__main__":
    main()










