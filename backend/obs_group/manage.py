# Standard Library Imports
import hashlib
import json
from datetime import date

# Third-Party Library Imports
import streamlit as st
import geopandas as gpd

# Local/Application-Specific Imports
# (No local/application-specific imports)



def initialize_observation_groups(num_sets):
    if 'observation_groups' not in st.session_state:
        st.session_state.observation_groups = [{} for _ in range(num_sets)]


def update_observation_group(index, **kwargs):
    if 0 <= index < len(st.session_state.observation_groups):
        st.session_state.observation_groups[index].update(kwargs)


def get_observation_group(index):
    if 0 <= index < len(st.session_state.observation_groups):
        return st.session_state.observation_groups[index]
    return {}


def get_all_observation_groups():
    return [group for group in st.session_state.observation_groups if group]


def are_observation_groups_valid():
    all_groups = get_all_observation_groups()

    if not all_groups:
        return False

    for group in all_groups:
        if ('aoi' not in group or
                not isinstance(group['aoi'], gpd.GeoDataFrame) or
                'date' not in group or
                'ground_truth' not in group or
                not isinstance(group['ground_truth'], gpd.GeoDataFrame)):
            return False

    return True


def are_observation_groups_valid_and_covered():
    all_groups = get_all_observation_groups()

    if not all_groups:
        return False

    for group in all_groups:
        # Check if the group has all required attributes
        if ('aoi' not in group or
                not isinstance(group['aoi'], gpd.GeoDataFrame) or
                'date' not in group or
                'feature_image' not in group or
                'ground_truth' not in group or
                not isinstance(group['ground_truth'], gpd.GeoDataFrame)):
            return False

    return True


def hash_observation_groups():
    # Extract relevant fields from the observation groups
    relevant_data = [
        {
            'date': group.get('date').isoformat() if isinstance(group.get('date'), date) else group.get('date'),
            'label': group.get('label')
        }
        for group in st.session_state.observation_groups
    ]
    # Convert to JSON string
    observation_groups_json = json.dumps(relevant_data, sort_keys=True)
    # Generate and return MD5 hash
    return hashlib.md5(observation_groups_json.encode()).hexdigest()


def hash_single_observation_group(group):
    # Extract relevant fields from the observation group
    relevant_data = {
        'date': group.get('date').isoformat() if isinstance(group.get('date'), date) else group.get('date'),
        'label': group.get('label')
    }
    # Convert to JSON string
    observation_group_json = json.dumps(relevant_data, sort_keys=True)
    # Generate and return MD5 hash
    return hashlib.md5(observation_group_json.encode()).hexdigest()
