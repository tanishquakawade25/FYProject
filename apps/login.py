import streamlit as st

def app():
    st.title("Login Page")
    username = st.text_input("👨🏻‍💼 Username")
    password = st.text_input("🔑 Password", type='password',)

    if st.button("Login"):
        if username == "user" and password == "1234":
            st.session_state.logged_in = True
            st.session_state.page = "home"
            st.success("✅ Login Successful!")
        else:
            st.error("❌ Invalid credentials!")




