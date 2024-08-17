# backend/session_state.py
import streamlit as st


def initialize_session_state():
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = ['VV', 'VH', 'DEM', 'slope', 'aspect']

    if 'all_features' not in st.session_state:
        st.session_state.all_features = ['VV', 'VH', 'angle', 'VV2_VH2', 'VV2_plus_VH2', 'VV_plus_VH', 'DEM', 'slope',
                                         'aspect']

    # Add more session state initializations as needed
    # if 'another_variable' not in st.session_state:
    #     st.session_state.another_variable = default_value
