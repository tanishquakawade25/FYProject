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




# import streamlit as st
# from apps import login, home

# st.set_page_config(
#     page_title="Stock Market",
#     page_icon="chart_with_upwards_trend",
# )

# def main():
#     st.title("Welcome to the Stock Market App")

#     # Initialize session state for sign-up form fields
#     if "signup_email" not in st.session_state:
#         st.session_state.signup_email = ""
#     if "signup_password" not in st.session_state:
#         st.session_state.signup_password = ""
#     if "signup_confirm_password" not in st.session_state:
#         st.session_state.signup_confirm_password = ""
#     if "current_page" not in st.session_state:
#         st.session_state.current_page = "home"

#     # Handle page navigation
#     if st.session_state.current_page == "home":
#         show_home()
#     elif st.session_state.current_page == "signup":
#         show_signup()
#     elif st.session_state.current_page == "login":
#         show_login()

# def show_home():
#     st.subheader("Home Page")
#     if st.button("Sign-up"):
#         st.session_state.current_page = "signup"
#     if st.button("Login"):
#         st.session_state.current_page = "login"

# def show_signup():
#     st.subheader("Sign-up Page")
#     st.session_state.signup_email = st.text_input("Enter Your Email", value=st.session_state.signup_email)
#     st.session_state.signup_password = st.text_input("Enter Password", type="password", value=st.session_state.signup_password)
#     st.session_state.signup_confirm_password = st.text_input(
#         "Confirm Password", type="password", value=st.session_state.signup_confirm_password
#     )

#     if st.button("Create Account"):
#         if st.session_state.signup_password == st.session_state.signup_confirm_password:
#             try:
#                 from firebase_admin import auth  # Import here to avoid circular issues
#                 user = auth.create_user(email=st.session_state.signup_email, password=st.session_state.signup_password)
#                 st.success(f"✅ Successfully created user: {user.uid}")
#             except Exception as e:
#                 st.error(f"Error creating user: {e}")
#         else:
#             st.error("❌ Passwords do not match.")

#     if st.button("Back to Home"):
#         st.session_state.current_page = "home"

# def show_login():
#     st.subheader("Login Page")
#     email = st.text_input("📧 Email", value=st.session_state.get("login_email", ""))
#     password = st.text_input("🔑 Password", type="password", value=st.session_state.get("login_password", ""))

#     if st.button("Login"):
#         try:
#             from firebase_admin import auth
#             user = auth.get_user_by_email(email)
#             st.success(f"✅ Login Successful! Welcome, {email}!")
#             st.session_state.logged_in = True
#             st.session_state.current_page = "home"
#         except Exception as e:
#             st.error(f"Login failed: {e}")

#     if st.button("Back to Home"):
#         st.session_state.current_page = "home"
        
        

# if __name__ == "__main__":
#     main()
