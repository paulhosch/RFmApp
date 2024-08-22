import streamlit as st


def home():
    st.title("Welcome to RFmApp")

    import ee
    import json

    # Load and initialize Earth Engine credentials
    try:
        service_account_info = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(email=service_account_info['client_email'],
                                                   key_data=service_account_info['private_key'])
        ee.Initialize(credentials)
        st.success("Successfully initialized Google Earth Engine!")
    except Exception as e:
        st.error(f"Failed to initialize Earth Engine: {e}")

    # Test Earth Engine functionality
    try:
        # Example: Get metadata of a specific image
        image = ee.Image('LANDSAT/LC08/C01/T1_TOA/LC08_044034_20140318')
        image_info = image.getInfo()

        # Display image metadata in Streamlit
        st.write("Landsat 8 Image Metadata:", image_info)
    except Exception as e:
        st.error(f"Error retrieving image metadata: {e}")

    # Optionally, add more Earth Engine operations to test here
