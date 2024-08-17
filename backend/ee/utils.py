from datetime import timedelta

import geopandas as gpd
import ee
import streamlit as st

from backend.obs_group import *


def convert_observation_groups_to_ee():
    all_groups = get_all_observation_groups()
    total_groups = len(all_groups)

    # Create a progress bar
    progress_bar = st.progress(0)


    for group_index, group in enumerate(all_groups):
        # Update progress bar
        progress = int((group_index + 1) / total_groups * 100)
        progress_bar.progress(progress)


        if 'aoi' in group and isinstance(group['aoi'], gpd.GeoDataFrame) and 'date' in group:
            aoi = group['aoi']
            observation_date = group['date']

            aoi_geojson = aoi.__geo_interface__
            aoi_ee = ee.Geometry.Polygon(aoi_geojson['features'][0]['geometry']['coordinates'])

            start_date_ee = ee.Date(observation_date.isoformat())
            end_date_ee = ee.Date((observation_date + timedelta(days=1)).isoformat())

            # Update the observation group with the new EE objects
            update_observation_group(group_index,
                                                  aoi_ee=aoi_ee,
                                                  start_date_ee=start_date_ee,
                                                  end_date_ee=end_date_ee)

        else:
            st.warning(f"Group {group_index + 1} does not have valid AOI or date. Skipping.")

    # Clear the progress bar and status message
    progress_bar.empty()

def get_local_min_max(image, aoi):
    """Calculate local min and max values for an image within an AOI."""
    min_max = image.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=aoi,
        scale=100,  # Adjust this scale as needed
        maxPixels=1e6
    )
    return min_max.getInfo()
