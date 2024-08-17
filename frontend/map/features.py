import streamlit as st
from .utils import plot_ee_image
from static.vis_params import feature_vis_params

def plot_feature_maps(selected_label, features):
    # Find the selected observation group
    selected_group = next(group for group in st.session_state.observation_groups if group['label'] == selected_label)
    feature_image = selected_group['feature_image']

    progress_bar = st.progress(0)
    cols = st.columns(len(features))

    figures = {}
    for idx, feature in enumerate(features):
        progress = int((idx + 1) / len(features) * 100)
        progress_bar.progress(progress, text=f'Plotting map {idx + 1} of {len(features)}: {feature}')

        with cols[idx]:
            label = f"{feature} - {selected_group['label']}"

            with st.spinner(f"Plotting Map of {feature} inside {selected_group['label']}..."):
                image = feature_image.select(feature)
                vis_params = feature_vis_params.get(feature, {}).copy()

                if 'min' not in vis_params:
                    vis_params['min'] = float(selected_group['feature_min_max'][feature]['min'])
                if 'max' not in vis_params:
                    vis_params['max'] = float(selected_group['feature_min_max'][feature]['max'])

                figure = plot_ee_image(image, vis_params, label, _discrete=False)
                figures[feature] = figure

    progress_bar.empty()
    st.session_state['feature_maps'] = figures