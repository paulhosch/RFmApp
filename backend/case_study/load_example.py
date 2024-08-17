import os
from datetime import datetime

import geopandas as gpd
import streamlit as st

from backend.case_study import NUM_SETS, EXAMPLE_FOLDERS
from backend.obs_group import initialize_observation_groups, update_observation_group


def load_example():
    initialize_observation_groups(NUM_SETS)
    for i, folder in enumerate(EXAMPLE_FOLDERS):
        if i >= NUM_SETS:
            break

        folder_path = os.path.join("example_data", folder)

        # Load AOI and Ground Truth
        aoi = gpd.read_file(os.path.join(folder_path, "aoi.shp"))
        ground_truth = gpd.read_file(os.path.join(folder_path, "ground_truth.shp"))

        # Set observation date and label from folder name
        date_str, label = folder.split("_", 3)[:3], folder.split("_", 3)[3]
        date = datetime.strptime("_".join(date_str), "%Y_%m_%d").date()

        update_observation_group(i, aoi=aoi, ground_truth=ground_truth, label=label, date=date)
        st.success("Example data initialized successfully!")
