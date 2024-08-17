import os
import streamlit as st
from tempfile import TemporaryDirectory
import geopandas as gpd

from backend.obs_group import update_observation_group

def handle_file_upload(uploaded_files, key, group_index):
    if uploaded_files is not None and len(uploaded_files) > 0:
        with TemporaryDirectory() as tmpdir:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(tmpdir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            required_files = [".shp", ".shx", ".dbf"]
            uploaded_filenames = [uploaded_file.name for uploaded_file in uploaded_files]

            if all(any(fname.endswith(ext) for fname in uploaded_filenames) for ext in required_files):
                try:
                    shapefile_path = next(fname for fname in uploaded_filenames if fname.endswith(".shp"))
                    shapefile_gdf = gpd.read_file(os.path.join(tmpdir, shapefile_path))

                    update_key = 'aoi' if key.startswith('aoi') else 'ground_truth'
                    update_observation_group(group_index, **{update_key: shapefile_gdf})

                    st.success(f"{update_key.capitalize()} loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading {update_key}: {str(e)}")
            else:
                st.error(f"Please upload all necessary shapefile components (.shp, .shx, .dbf).")