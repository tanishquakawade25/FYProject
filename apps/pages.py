import streamlit as st
import base64

# Path to your logo image
LOGO_URL_LARGE = "C:/Users/hp/Desktop/FYProject/ki.jpg"

# Function to encode the image in Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Encode the image
logo_base64 = get_base64_image(LOGO_URL_LARGE)

# Custom CSS to display the logo more to the right side
st.markdown(
    f"""
    <style>
    .logo-container {{
        position: absolute;
        top: 1px;
        right: 5px;  /* Adjust distance from the right edge */
        transform: translateX(300%);  /* Moves it further to the right (center horizontally) */
        z-index: 1;
    }}
    .logo-container img {{
        width: 120px;  /* Adjust the size */
        height: auto;
    }}
    </style>
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_base64}">
    </div>
    """,
    unsafe_allow_html=True
)
