from pathlib import Path

import streamlit as st
import hydralit_components as hc

import pages as pages
from backend.utils import initialize_session_state

# Define the path to the project root and construct the logo path
project_root = Path(__file__).parent
logo_path = project_root / "static" / "images" / "logo.svg"

# Set up Streamlit page configuration with custom favicon
st.set_page_config(
    page_title="RFmApp",
    page_icon=str(logo_path),  # Use your logo as favicon
    initial_sidebar_state="collapsed",
    layout="wide"
)

# Initialize global session state variables
initialize_session_state()


# CSS to hide the sidebar and its open icon
hide_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
        button[kind="header"] {display: none;}
        section[data-testid="stSidebar"] {display: none;}
    </style>
"""

# Inject the CSS to hide the sidebar
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

st.write('<style>div.block-container{padding:0rem;}</style>', unsafe_allow_html=True)

# Define the menu data for Hydralit navbar
menu_data = [
    {'icon': "fas fa-home fa-icon", 'label': "Home"},
    {'icon': "fas fa-map-marked-alt fa-icon", 'label': "Case Study"},
    {'icon': "fas fa-braille fa-icon", 'label': "Sampling"},
    {'icon': "fas fa-layer-group fa-icon", 'label': "Features"},
    {'icon': "fas fa-sliders-h fa-icon", 'label': "Tuning"},
]

# Create the Hydralit navbar
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme={'txc_inactive': '#FFFFFF'},
    #home_name='Home',
    #hide_streamlit_markers=False,
    sticky_nav=True,
    sticky_mode='pinned',
)

# Render the selected page based on user navigation
if menu_id == "Home":
    pages.home()
elif menu_id == "Case Study":
    pages.case_study()
elif menu_id == "Features":
    pages.features()
elif menu_id == "Sampling":
    pages.sampling()
elif menu_id == "Tuning":
    pages.tuning()

# You can add the logo to your pages separately, for example:
# st.sidebar.image(str(logo_path), use_column_width=True)