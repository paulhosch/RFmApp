import streamlit as st
from backend.ee import initialize_rfmapp


def home():
    initialize_rfmapp()
