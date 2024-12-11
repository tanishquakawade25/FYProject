from config.firebase_setup import get_db
import streamlit as st
db = get_db()
st.write(db)
