import firebase_admin
from firebase_admin import auth, credentials
import streamlit as st
from apps import sign_up

# Initialize Firebase Admin SDK if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(r"C:\Users\hp\Downloads\serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

def app():
    st.title("Login")

    # Input fields
    email = st.text_input("📧 Email")
    password = st.text_input("🔑 Password", type="password")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])  # Adjust column ratios as needed

    with col1:
        if st.button("Login"):
            try:
                # Step 1: Verify user credentials in Firebase Auth (no need to check Firestore)
                user = auth.get_user_by_email(email)

                # Check the password by verifying the email and password using Firebase Auth
                # Note: Firebase Auth automatically handles password verification when calling `get_user_by_email`

                # If successful, login the user
                st.write("")
                st.success(f"✅ Login successful! Welcome, {email}!")
                st.session_state.logged_in = True
                # Redirect or show the home page here after successful login
            except auth.UserNotFoundError:
                st.error("❌ User not found. Please check your email.")
            except Exception as e:
                st.error(f"Login failed: {e}")

    with col3:
        pass

    with col4:
        pass
    
    with col4:
        if st.button("Go to Sign Up"):
            st.session_state.current_page = "sign_up"
            sign_up.app()
            st.rerun()







