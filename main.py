from pathlib import Path

import streamlit as st
from streamlit_navigation_bar import st_navbar

import pages as pages
from static.styles import nav_style, nav_options
from backend.utils import initialize_session_state

# Set up Streamlit page configuration
st.set_page_config(initial_sidebar_state="collapsed", layout="wide")

# Initialize global session state variables
initialize_session_state()

# Define the path to the project root and construct the logo path
project_root = Path(__file__).parent
logo_path = project_root / "static" / "images" / "logo.svg"

NAMES = ["Home", "Case Study", "Sampling", "Features", "Tuning"]

# Create the navigation bar
page = st_navbar(
    NAMES,
    logo_path=str(logo_path),  # Convert the Path object to string as required by st_navbar
    styles=nav_style,
    options=nav_options,
)

# Render the selected page based on user navigation
if page == "Home":
    pages.home()
elif page == "Case Study":
    pages.case_study()
elif page == "Features":
    pages.features()
elif page == "Sampling":
    pages.sampling()
elif page == "Tuning":
    pages.tuning()