import numpy as np
import streamlit as st
import ee


def convert_sampled_points_to_fc(sampled_points_df):
    features = sampled_points_df.apply(lambda row: ee.Feature(
        ee.Geometry.Point([row['lon'], row['lat']]),
        {'class': 1 if row['class'] == 1 else 0}
    ), axis=1)
    return ee.FeatureCollection(features.tolist())


@st.cache_data()
def get_group_X_y(_group, group_hash, sampled_points_df):
    aoi_ee = _group['aoi_ee']
    image = _group['feature_image']
    all_features = st.session_state.all_features

    features = []
    labels = []

    # Convert sampled points to ee.FeatureCollection
    sampled_points_ee = convert_sampled_points_to_fc(sampled_points_df)

    # Sample the image vector
    sampled_data = image.select(all_features).sampleRegions(
        collection=sampled_points_ee,
        properties=['class'],
        scale=10
    ).getInfo()

    # Extract features and labels
    for feature in sampled_data['features']:
        properties = feature['properties']

        if all(var in properties for var in all_features):
            feature_vector = [properties[var] for var in all_features]
            features.append(feature_vector)
            labels.append(properties['class'])

    X = np.array(features)  # input variable values
    y = np.array(labels)  # class label values

    return X, y
