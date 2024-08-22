import streamlit as st
from backend.ee import initialize_rfmapp


def home():
    initialize_rfmapp()

    # Add a button to the Streamlit app
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")