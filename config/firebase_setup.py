import sys
sys.path.append('C:/Users/hp/Desktop/FYProject/config')

import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(r"C:\Users\hp\Downloads\serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

# Firestore client function
def get_db():
    """Return the Firestore client."""
    return firestore.client()
