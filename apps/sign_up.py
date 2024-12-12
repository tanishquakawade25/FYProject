import firebase_admin
from firebase_admin import credentials, auth
import streamlit as st
from apps import login

# Initialize Firebase Admin SDK if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(r"C:\Users\hp\Downloads\serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

def app():
    st.title("Sign Up")

    # Input fields
    email = st.text_input("📧 Email")
    password = st.text_input("🔑 Password", type="password")
    confirm_password = st.text_input("🔒 Confirm Password", type="password")

    if st.button("Create Account"):
        if password != confirm_password:
            st.error("❌ Passwords do not match.")
        else:
            try:
                # Create user in Firebase
                user = auth.create_user(email=email, password=password)
                st.success(f"✅ Account created successfully! User ID: {user.uid}")
                
                if st.button("Go to Login"):
                    st.session_state.current_page = "login"
                    login.app()
                    st.rerun()
                    
                    
            except Exception as e:
                st.error(f"Error creating account: {e}")
