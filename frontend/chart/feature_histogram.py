import streamlit as st
import plotly.express as px
import ee
import pandas as pd

from backend.obs_group import *

@st.cache_data()
def get_cached_histogram(_feature_image, _aoi, feature, group_hash):
    # Get the histogram for this feature/band
    histogram_result = _feature_image.select(feature).reduceRegion(
        reducer=ee.Reducer.histogram(maxBuckets=50),
        geometry=_aoi,
        scale=50,
        maxPixels=1e9
    ).getInfo()

    hist = histogram_result[feature]

    return hist
def get_histogram_data(observation_groups, group_hash, features):
    all_data = []
    total_groups = len(observation_groups)
    total_features = len(features)
    group_progress = st.progress(0)

    for j, group in enumerate(observation_groups):
        feature_image = group['feature_image']
        aoi = group['aoi_ee']
        label = group['label']
        group_progress.progress((j + 1) / total_groups, text=f"Getting Histograms inside {label} ...")
        group_hash = hash_single_observation_group(group)

        feature_progress = st.progress(0)
        for i, feature in enumerate(features):
            feature_progress.progress((i + 1) / total_features, text=f"For {feature} band ...")

            histogram = get_cached_histogram(feature_image, aoi, feature, group_hash)

            # Extract bucket means and histogram counts
            bucket_means = histogram.get('bucketMeans', [])
            counts = histogram.get('histogram', [])

            # Create a DataFrame for this histogram
            df = pd.DataFrame({
                'Feature': feature,
                'Group': label,
                'Value': bucket_means,
                'Frequency': counts
            })

            # Repeat each value based on its count
            df = df.loc[df.index.repeat(df['Frequency'].astype(int))].reset_index(drop=True)

            all_data.append(df[['Feature', 'Group', 'Value']])

        feature_progress.empty()

    group_progress.empty()


    # Combine all data into a single DataFrame
    df = pd.concat(all_data, ignore_index=True)
    return df

@st.cache_resource()
def create_ridgeline_plot(_data, group_hash, features):
    with st.spinner('Plotting Histogram Data'):
        categories = _data['Feature'].unique()
        groups = _data['Group'].unique()

        viridis_colors = px.colors.sample_colorscale('Viridis', [i/(len(groups)-1) for i in range(len(groups))])
        group_colors = {group: viridis_colors[i] for i, group in enumerate(groups)}

        # Create the ridgeline plot with overlaid groups per category
        fig = px.violin(_data,
                        x="Value",
                        y="Feature",
                        color="Group",
                        violinmode='overlay',
                        hover_data=_data.columns,
                        category_orders={
                            "category": categories[::-1]}, # Reverse category order for ridgeline effect
                            color_discrete_map=group_colors  # Apply Viridis colors to the groups
                        )

        # Update layout for better visualization
        fig.update_traces(side='positive', width=3, points=False)
        fig.update_layout(
            xaxis_showgrid=False,
            xaxis_zeroline=False,
            height=100 * len(categories) + 100  # Adjust height based on number of categories
        )

    return fig