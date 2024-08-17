import streamlit as st
import ee

from backend.obs_group import update_observation_group


@st.cache_data()
def get_image_min_max(_group, group_hash, feature):
    image = _group['feature_image'].select(feature)
    # Calculate min and max
    min_max = image.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=_group['aoi_ee'],
        scale=50,  # Adjust as needed
        maxPixels=1e9
    ).getInfo()

    return min_max


def add_feature_min_max(_i, _group, features, group_hash):
    min_max_dict = {}

    for i, feature in enumerate(features):
        min_max = get_image_min_max(_group, group_hash, feature)

        min_max_dict[feature] = {
            'min': min_max[f'{feature}_min'],
            'max': min_max[f'{feature}_max']
        }

    # Update the group's session state
    update_observation_group(_i, feature_min_max=min_max_dict)
    return
