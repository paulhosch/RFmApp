import streamlit as st
import ee
import json
import pandasql as psql
import os


def ee_init():
    # Load the service account credentials from Streamlit secrets
    service_account_info = {
        "type": st.secrets["gcp_service_account"]["type"],
        "project_id": st.secrets["gcp_service_account"]["project_id"],
        "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
        "private_key": st.secrets["gcp_service_account"]["private_key"],
        "client_email": st.secrets["gcp_service_account"]["client_email"],
        "client_id": st.secrets["gcp_service_account"]["client_id"],
        "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
        "token_uri": st.secrets["gcp_service_account"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
        "universe_domain": st.secrets["gcp_service_account"]["universe_domain"]
    }

    # Initialize Earth Engine with the service account credentials
    credentials = ee.ServiceAccountCredentials(email=service_account_info['client_email'],
                                               key_data=json.dumps(service_account_info))
    ee.Initialize(credentials)

    # Example to fetch and display a value from the Earth Engine servers
    return ee.String('Earth Engine API Connected').getInfo()



def home():
    col1, col2 = st.columns([1,4])
    col1.title("RFmApp")

    tut_cnt = col1.container(border=True)
    tut_cnt.header("Tutorial")
    img_cnt = col2.container(border=True)
    img_cnt.image(os.path.join('static', 'images', 'home.png'), use_column_width=True)



    with col1:

        with st.spinner('Connecting Google Earth Engine API ...'):
            conn_test = ee_init()
            st.toast(conn_test)


        if st.button("Clear Cache and Restart App"):
            st.cache_data.clear()
            st.rerun()
            st.toast("Cache cleared!", icon=":material/check_circle:")

        st.write('Workaround for some cache related Bugs')

