# standard library imports
from datetime import datetime
import os
from tempfile import TemporaryDirectory

# related third party imports
import streamlit as st
import ee
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import unary_union
import hydralit_components as hc

# local app specific imports
from backend.obs_group import *
from backend.case_study import load_example, handle_file_upload, NUM_SETS
from backend.ee import convert_observation_groups_to_ee, add_feature_image_to_group
from frontend.map import plot_all_aois
from backend.case_study import NUM_SETS, EXAMPLE_FOLDERS


def merge_hydrography_with_ground_truth(ground_truth_gdf, hydrography_gdf):
    # Ensure both GeoDataFrames have the same CRS
    if ground_truth_gdf.crs != hydrography_gdf.crs:
        hydrography_gdf = hydrography_gdf.to_crs(ground_truth_gdf.crs)

    # Reset indices to avoid warnings
    ground_truth_gdf = ground_truth_gdf.reset_index(drop=True)
    hydrography_gdf = hydrography_gdf.reset_index(drop=True)

    # Combine all geometries from both GeoDataFrames
    all_geometries = list(ground_truth_gdf.geometry) + list(hydrography_gdf.geometry)

    # Merge geometries
    merged_geometry = unary_union(all_geometries)

    # Create a new GeoDataFrame with the merged geometry
    merged_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=ground_truth_gdf.crs)

    # Preserve attributes from ground truth
    for col in ground_truth_gdf.columns:
        if col != 'geometry':
            merged_gdf[col] = ground_truth_gdf[col].iloc[0]  # Assuming we want to keep the first value

    return merged_gdf


def handle_hydrography_upload(uploaded_files, tmpdir):
    if uploaded_files and len(uploaded_files) > 0:
        required_files = [".shp", ".shx", ".dbf"]
        uploaded_filenames = [uploaded_file.name for uploaded_file in uploaded_files]

        if all(any(fname.endswith(ext) for fname in uploaded_filenames) for ext in required_files):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(tmpdir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            shapefile_path = next(fname for fname in uploaded_filenames if fname.endswith(".shp"))
            return gpd.read_file(os.path.join(tmpdir, shapefile_path))

    return None

def case_study():
    initialize_observation_groups(NUM_SETS)

    col1, col2, col3, = st.columns([1, 3, 2])

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

                st.write("**Hydrography**")
                hydro_files = st.file_uploader(
                    f'Hydrography Set {i + 1}',
                    type=["shp", "dbf", "shx", "prj"],
                    accept_multiple_files=True,
                    key=f'hydro_files_{i + 1}'
                )

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

                # Process and merge hydrography data
                with TemporaryDirectory() as tmpdir:
                    hydrography_gdf = handle_hydrography_upload(hydro_files, tmpdir)
                    if hydrography_gdf is not None:
                        ground_truth_gdf = get_observation_group(i).get('ground_truth')
                        if ground_truth_gdf is not None:
                            try:
                                merged_gdf = merge_hydrography_with_ground_truth(ground_truth_gdf, hydrography_gdf)
                                update_observation_group(i, ground_truth=merged_gdf)
                                st.toast(f"Hydrography data merged with ground truth for Group {i + 1}")
                            except Exception as e:
                                st.error(f"Error merging hydrography data for Group {i + 1}: {str(e)}")
                        else:
                            st.warning(
                                f"Ground truth data not available for Group {i + 1}. Upload ground truth before hydrography.")
                    elif hydro_files:
                        st.error(
                            f"Error processing hydrography files for Group {i + 1}. Ensure all required files are uploaded.")

    if col1.button("Initialize Example", type="primary", use_container_width=True):
            load_example()
            # Load and merge example hydrography data
            for i, folder in enumerate(EXAMPLE_FOLDERS):
                if i >= NUM_SETS:
                    break
                folder_path = os.path.join("example_data", folder)
                hydrography_path = os.path.join(folder_path, "hydrographyA.shp")
                if os.path.exists(hydrography_path):
                    hydrography_gdf = gpd.read_file(hydrography_path)
                    ground_truth_gdf = get_observation_group(i).get('ground_truth')
                    if ground_truth_gdf is not None:
                        merged_gdf = merge_hydrography_with_ground_truth(ground_truth_gdf, hydrography_gdf)
                        update_observation_group(i, ground_truth=merged_gdf)
            st.toast("Example data initialized and hydrography merged successfully!")

    with col2:
        observation_groups = get_all_observation_groups()
        valid_groups = [
            (group['aoi'], group['ground_truth'], group['label'], group['date'])
            for group in observation_groups
            if 'aoi' in group and 'ground_truth' in group and isinstance(group['aoi'], gpd.GeoDataFrame) and isinstance(group['ground_truth'], gpd.GeoDataFrame)
        ]

        if valid_groups:
            try:
                with hc.HyLoader('', hc.Loaders.pretty_loaders, index=3):

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

        convert_observation_groups_to_ee()

        progress_bar = st.progress(0)
        for i, group in enumerate(observation_groups):
            progress = int((i + 1) / len(observation_groups)*100)
            progress_bar.progress(progress, text=f"Processing {group['label']} ({group['date']})")

            group_hash = hash_single_observation_group(group)
            add_feature_image_to_group(i, group, group_hash)

        progress_bar.empty()