import streamlit as st
import geemap
import ee
import hydralit_components as hc


@st.cache_data(show_spinner=False)
def initialize_earth_engine_cached(token_name="EARTHENGINE_TOKEN"):
    try:
        # Initialize Earth Engine
        geemap.ee_initialize(token_name=token_name)

        # Perform a simple operation to check connection
        ee.Number(1).getInfo()  # Fetches a constant value from EE servers

        # Display success message
        st.success("GEE initialized!")

        return True  # Return True if everything works
    except Exception as e:
        st.error(f"Error initializing Earth Engine: {e}")
        return False  # Return False if there is an error


def initialize_rfmapp():
    with hc.HyLoader('Initializing RFmApp...', hc.Loaders.pretty_loaders, index=3):
        initialize_earth_engine_cached(token_name="EARTHENGINE_TOKEN")

    st.success("RFmApp ready!")