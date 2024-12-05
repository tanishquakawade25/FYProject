# import streamlit as st

# def app():
#     st.title("Login")

#     # Assign unique keys to the text_input elements
#     username = st.text_input("Username", key="login_username")
#     password = st.text_input("Password", type="password", key="login_password")

#     if st.button("Login"):
#         if username == "admin" and password == "1234":  # Example credentials
#             st.success("Logged in successfully!")
#             st.session_state.logged_in = True
#             st.session_state.page = "home"
#             st.success("Login Successful!")
#         else:
#             st.error("Invalid username or password.")





import streamlit as st

def app():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        if username == "user" and password == "1234":
            st.session_state.logged_in = True
            st.session_state.page = "home"
            st.success("Login Successful!")
        else:
            st.error("Invalid credentials!")




