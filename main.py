from pathlib import Path

import streamlit as st
from streamlit_navigation_bar import st_navbar

import pages as pages
from static.styles import nav_style, nav_options
from backend.utils import initialize_session_state

st.set_page_config(initial_sidebar_state="collapsed", layout="wide")

#Initliaize some globale states
initialize_session_state()

# Construct the correct path to the logo
project_root = Path(__file__).parent
logo_path = project_root / "static" / "images" / "logo.svg"



# Create the navigation bar
page = st_navbar(
    pages.NAMES,
    logo_path=str(logo_path),
    styles=nav_style,
    options=nav_options,
)

# Render the selected page
if page == "Home":
    pages.home()

if page == "Case Study":
    pages.case_study()
