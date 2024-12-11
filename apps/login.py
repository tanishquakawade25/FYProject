import sys
sys.path.append('C:/Users/hp/Desktop/FYProject')


import firebase_admin
from firebase_admin import auth, credentials, firestore
import streamlit as st
from apps import sign_up

# Initialize Firebase Admin SDK if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(r"C:\Users\hp\Downloads\serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()

def app():
    st.title("Login")

    # Input fields
    email = st.text_input("📧 Email")
    password = st.text_input("🔑 Password", type="password")

    col1, col2 = st.columns([1,1])  # Adjust column ratios as needed

    with col1:
        if st.button("Login"):
            try:
                # Verify user credentials in Firebase Auth
                user = auth.get_user_by_email(email)

                # Additional authorization check in Firestore
                if is_email_authorized(email):
                    st.success(f"✅ Login successful! Welcome, {email}!")
                    st.session_state.logged_in = True
                    st.experimental_rerun()  # Redirect to home page
                else:
                    st.error("❌ Unauthorized email.")
            except auth.UserNotFoundError:
                st.error("❌ User not found. Please check your email.")
            except Exception as e:
                st.error(f"Login failed: {e}")

    with col2:
        if st.button("Go to Sign Up"):
            st.session_state.current_page = "sign_up"
            sign_up.app()
            st.experimental_rerun()

def is_email_authorized(email):
    """Check if the email is authorized in Firestore."""
    try:
        doc_ref = db.collection('authorized_users').document('emails')
        authorized_emails = doc_ref.get().to_dict()
        if authorized_emails and email in authorized_emails.get('list', []):
            return True
        return False
    except Exception as e:
        st.error(f"Error checking authorization: {e}")
        return False








# import streamlit as st
# import firebase_admin
# from firebase_admin import credentials, auth, firestore

# # Initialize Firebase Admin SDK
# if not firebase_admin._apps:
#     cred = credentials.Certificate(r"C:/Users/hp/Downloads/serviceAccountKey.json")  # Use raw string for path
#     firebase_admin.initialize_app(cred)

# db = firestore.client()

# # Function to check if an email is authorized
# def is_email_authorized(email, db):
#     try:
#         # Access Firestore document
#         doc_ref = db.collection('authorized_users').document('emails')
#         authorized_emails = doc_ref.get().to_dict()  # Fetch email list
#         if authorized_emails and email in authorized_emails.get('list', []):
#             return True
#         return False
#     except Exception as e:
#         st.error(f"Error checking authorization: {e}")
#         return False

# def app():
#     st.title("Login First")
    
#     # Input fields for email and password
#     email = st.text_input("📧 Email")
#     password = st.text_input("🔑 Password", type='password')

#     if st.button("Login"):
#         try:
#             # Verify email using Firebase Authentication
#             user = auth.get_user_by_email(email)
#             if user and is_email_authorized(email, db):
#                 # Login successful
#                 st.session_state.logged_in = True
#                 st.session_state.page = "home"
#                 st.success(f"✅ Login Successful! Welcome, {email}!")
#             else:
#                 st.error("❌ Unauthorized email or invalid credentials.")
#         except Exception as e:
#             st.error(f"Login failed: {e}")


# # For standalone testing
# if __name__ == "__main__":
#     app()
