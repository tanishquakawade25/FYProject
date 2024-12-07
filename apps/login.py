# import streamlit as st

# def app():
#     st.title("Login First")
#     username = st.text_input("👨🏻‍💼 Username")
#     password = st.text_input("🔑 Password", type='password',)

#     if st.button("Login"):
#         if username == "user" and password == "1234":
#             st.session_state.logged_in = True
#             st.session_state.page = "home"
#             st.success("✅ Login Successful!")
#         else:
#             st.error("❌ Invalid credentials!")





import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate(r"C:/Users/hp/Downloads/serviceAccountKey.json")  # Use raw string for path
    firebase_admin.initialize_app(cred)
    db = firestore.client()

# Function to check if an email is authorized
def is_email_authorized(email, db):
    try:
        # Access Firestore document
        doc_ref = db.collection('authorized_users').document('emails')
        authorized_emails = doc_ref.get().to_dict()  # Fetch email list
        if authorized_emails and email in authorized_emails.get('list', []):
            return True
        return False
    except Exception as e:
        st.error(f"Error checking authorization: {e}")
        return False

def app():
    st.title("Login First")
    
    # Input fields for email and password
    email = st.text_input("📧 Email")
    password = st.text_input("🔑 Password", type='password')

    if st.button("Login"):
        try:
            # Verify email using Firebase Authentication
            user = auth.get_user_by_email(email)
            if user and is_email_authorized(email, db):
                # Login successful
                st.session_state.logged_in = True
                st.session_state.page = "home"
                st.success(f"✅ Login Successful! Welcome, {email}!")
            else:
                st.error("❌ Unauthorized email or invalid credentials.")
        except Exception as e:
            st.error(f"Login failed: {e}")

# For standalone testing
if __name__ == "__main__":
    app()
