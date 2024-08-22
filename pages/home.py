import streamlit as st
from backend.ee import initialize_rfmapp


def home():
    st.title("RFmApp")
    st.write('Web Application for Flood Mapping using Random Forest and Synthetic Aperture Radar')
    initialize_rfmapp()

    # Add a button to the Streamlit app
    col1, col2 = st.columns([1,5])
    if col1.button("Clear Cache and Restart App"):
        st.cache_data.clear()
        st.rerun()
        st.toast("Cache cleared!", icon=":material/check_circle:")


    col2.write('Workaround for some cache related Bugs')