# standard library imports
from datetime import datetime

# related third party imports
import streamlit as st
import ee
import geopandas as gpd
import matplotlib.pyplot as plt

# local app specific imports
from backend.obs_group import *
from backend.case_study import load_example, handle_file_upload, NUM_SETS
from backend.ee import convert_observation_groups_to_ee, add_feature_image_to_group
from frontend.map import plot_all_aois

def case_study():
    initialize_observation_groups(NUM_SETS)

    col1, col2, col3, = st.columns([1, 4, 1])

    with col1:
        st.write("#### Define Multiple Observations Groups")
        st.write("Upload AOI and Ground Truth Shapefiles and Specify Observation Date!")
        for i in range(NUM_SETS):
            with st.expander(f"**Observation Group-{i+1}**"):
                st.write("**Area of Interest**")
                aoi_files = st.file_uploader(
                    f'AOI Set {i+1}',
                    type=["shp", "dbf", "shx", "prj"],
                    accept_multiple_files=True,
                    key=f'aoi_files_{i+1}'
                )
                handle_file_upload(aoi_files, f'aoi_{i+1}', i)

                st.write("**Ground Truth**")
                gt_files = st.file_uploader(
                    f'Ground Truth Set {i+1}',
                    type=["shp", "dbf", "shx", "prj"],
                    accept_multiple_files=True,
                    key=f'ground_truth_files_{i+1}'
                )
                handle_file_upload(gt_files, f'ground_truth_{i+1}', i)

                st.write("**Observation Date**")
                date = st.date_input(
                    f"Date Set {i+1}",
                    value=get_observation_group(i).get('date', datetime.now().date()),
                    key=f'date_input_{i+1}'
                )
                update_observation_group(i, date=date)

                st.write("**Site Label**")
                label = st.text_input(
                    f"Label Set {i+1}",
                    value=get_observation_group(i).get('label', f"Site {i+1}"),
                    key=f'label_input_{i+1}'
                )
                update_observation_group(i, label=label)

        if st.button("Initialize Example", type="primary", use_container_width=True):
            load_example()

    with col2:
        observation_groups = get_all_observation_groups()
        valid_groups = [
            (group['aoi'], group['ground_truth'], group['label'], group['date'])
            for group in observation_groups
            if 'aoi' in group and 'ground_truth' in group and isinstance(group['aoi'], gpd.GeoDataFrame) and isinstance(group['ground_truth'], gpd.GeoDataFrame)
        ]

        if valid_groups:
            try:
                figures = plot_all_aois(valid_groups)
                cols = st.columns(2)

                # Display AOI figures
                for i in range(0, len(figures) - 1):
                    col = cols[i % 2]
                    with col:
                        fig, title = figures[i]
                        st.pyplot(fig)
                        st.write(title)
                        plt.close(fig)

                # Display marker map in col3
                with col3:
                    if len(figures) > 0:
                        fig, title = figures[-1]
                        st.pyplot(fig)
                        st.write(title)
                        plt.close(fig)
                    else:
                        st.write("No marker map available.")
            except ValueError as e:
                st.error(f"Error plotting data: {str(e)}")
                st.write("Please ensure all data is properly loaded and formatted.")
        else:
            st.write("No valid data available for visualization. Please upload or initialize data.")


    if are_observation_groups_valid():
        observation_groups_hash = hash_observation_groups()

        # Initialize EE and convert group data
        @st.cache_data()
        def cache_init(hash):
            with st.spinner('Initializing GEE ...'):
                ee.Initialize()

            # Convert the data of all observation groups to EE Objects and add them to the session state dict
            convert_observation_groups_to_ee()
            return

        cache_init(observation_groups_hash)

        progress_bar = st.progress(0)
        for i, group in enumerate(observation_groups):
            progress = int((i + 1) / len(observation_groups)*100)
            progress_bar.progress(progress, text=f"Processing {group['label']} ({group['date']})")

            group_hash = hash_single_observation_group(group)
            add_feature_image_to_group(i, group, group_hash)

        progress_bar.empty()