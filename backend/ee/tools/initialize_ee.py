import streamlit as st
import geemap
import ee
import hydralit_components as hc

@st.cache_data(show_spinner=False)
def initialize_earth_engine(token_name="EARTHENGINE_TOKEN"):
    try:
        # Initialize Earth Engine
        geemap.ee_initialize(token_name=token_name)

        # Perform a simple operation to check connection
        ee.Number(1).getInfo()  # Fetches a constant value from EE servers

        return True  # Return True if everything works
    except Exception as e:
        return str(e)  # Return error message if there is an error


def initialize_rfmapp():
    with hc.HyLoader('Initializing RFmApp...', hc.Loaders.pretty_loaders, index=3):
        result = initialize_earth_engine(token_name="EARTHENGINE_TOKEN")

    if result is True:
        st.toast("GEE initialized!", icon=":material/check_circle:")
    else:
        st.error(f"Error initializing Earth Engine: {result}")

    st.toast("RFmApp ready!", icon=":material/check_circle:")