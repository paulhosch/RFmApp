# Standard Library Imports
# (No standard library imports)

# Third-Party Library Imports
import streamlit as st
import ee

# Local/Application-Specific Imports
from backend.obs_group import update_observation_group



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


def add_feature_min_max(i, group, features, group_hash):
    # Check if we have a cache for min-max values
    if 'min_max_cache' not in st.session_state:
        st.session_state.min_max_cache = {}

    min_max_dict = {}

    for feature in features:
        cache_key = (group_hash, feature)  # Create a unique key for this feature and group hash combination

        if cache_key in st.session_state.min_max_cache:
            # If we have cached values for this feature, use them
            min_max = st.session_state.min_max_cache[cache_key]
        else:
            # If not, calculate and cache the values for this feature
            min_max = get_image_min_max(group, group_hash, feature)
            st.session_state.min_max_cache[cache_key] = min_max

        min_max_dict[feature] = {
            'min': min_max[f'{feature}_min'],
            'max': min_max[f'{feature}_max']
        }

    # Update the group's session state
    update_observation_group(i, feature_min_max=min_max_dict)
    return min_max_dict

