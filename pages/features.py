# standard library imports
# related third party imports
import ee
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_option_menu import option_menu

# local app specific imports
from backend.obs_group import *
from backend.ee import add_feature_min_max, compute_pairwise_correlations_for_groups, get_feature_importance
from frontend.map import plot_feature_maps
from frontend.chart import (get_histogram_data, create_ridgeline_plot, plot_3d_correlation_scatter_with_heatmap,
                            plot_impurity_importances, plot_permutation_importances, plot_shap_values)
from static.styles import opt_menu_style, border_container


def features():
    #### Initialize ####
    st.write("### Feature Engineering")
    st.empty()  # Clear the UI

    if not are_observation_groups_valid_and_covered():
        st.error("Please specify the observation groups first.", icon="⚠")

    observation_groups = get_all_observation_groups()
    observation_group_labels = [group['label'] for group in observation_groups]
    observation_groups_hash = hash_observation_groups()

    col1, col2, col3, col4 = st.columns(4)

    #### Plot Maps ####
    with col1.form('spacial_distribution'):
        st.write("##### Spacial Distribution of Features")

        features_to_map = st.multiselect('Select Input Features',
                                         st.session_state.all_features,
                                         default=st.session_state.selected_features,
                                         label_visibility='collapsed')

        group_to_map = option_menu(
            None, observation_group_labels,
            icons=['map' for _ in observation_group_labels],
            menu_icon="cast", default_index=0, orientation="horizontal", styles=opt_menu_style
        )
        st.divider()

        map_features = st.form_submit_button(label='Plot Maps', use_container_width=True, type='primary')

    if map_features:
        # Get feature min_max
        progress_bar = st.progress(0)
        for i, group in enumerate(observation_groups):
            progress = int((i + 1) / len(observation_groups) * 100)
            map_group_hash = hash_single_observation_group(group)
            progress_bar.progress(progress, text=f"Calculating extrema for {group['label']} on {group['date']} ...")

            # Add min/max values
            add_feature_min_max(i, group, features_to_map, map_group_hash)

        progress_bar.empty()

        plot_feature_maps(group_to_map, features_to_map)

    if 'feature_maps' in st.session_state:
        with stylable_container(key='map_container', css_styles=border_container):
            st.write("##### Spacial Distribution of Features")
            features = st.session_state['feature_maps']
            cols = st.columns(len(features))
            for idx, (feature, figure) in enumerate(features.items()):
                with cols[idx]:
                    label = f'{feature}'
                    st.write(f"**<div style='text-align: center;'>{feature}</div>**", unsafe_allow_html=True)
                    st.pyplot(figure)

    #### Plot Ridgeline Histograms ####
    with col2.form('value_distribution'):
        st.write("##### Value Distribution of Features")
        features_to_plot = st.multiselect('Select Input Features',
                                          st.session_state.all_features,
                                          default=st.session_state.selected_features,
                                          label_visibility='collapsed')
        chart = option_menu(
            None, ['Ridgeline', 'Histogram', 'Boxplot'], icons=['graph-down', 'bar-chart', 'bar-chart-steps'],
            menu_icon="cast", default_index=0, orientation="horizontal", styles=opt_menu_style
        )

        st.divider()

        plot_features = st.form_submit_button(label='Plot Values', use_container_width=True, type='primary')
    if plot_features:
        if chart == 'Ridgeline':
            data = get_histogram_data(observation_groups, observation_groups_hash, features_to_plot)
            st.session_state.ridgeline_fig = create_ridgeline_plot(data, observation_groups_hash, features_to_plot)
        else:
            st.warning("Not implemented yet")

    if 'ridgeline_fig' in st.session_state:
        with stylable_container(key='value_container', css_styles=border_container):
            st.write("###### Value Distribution of Features")
            with st.spinner('Plotting ...'):
                st.plotly_chart(st.session_state.ridgeline_fig)

    #### Plot Correlation  ####
    with col3.form('correlation'):
        st.write("##### Global Correlation Analysis")
        features_to_analyse = st.multiselect('Select Input Features',
                                             st.session_state.all_features,
                                             default=st.session_state.selected_features,
                                             label_visibility='collapsed')

        reducer = option_menu(
            None, ['Pearson', 'Spearman', 'Difference'], icons=['graph-up', 'list-ol', 'info'],
            menu_icon="cast", default_index=0, orientation="horizontal", styles=opt_menu_style
        )
        st.divider()

        analyse_features = st.form_submit_button(label='Plot Correlation', use_container_width=True, type='primary')

    if analyse_features:

        if reducer == 'Difference':
            st.warning("Not implemented yet")
        else:
            if reducer == 'Pearson':
                ee_reducer = ee.Reducer.pearsonsCorrelation()
            if reducer == 'Spearman':
                ee_reducer = ee.Reducer.spearmansCorrelation()

            with st.spinner('Computing Correlation ...'):
                correlation = compute_pairwise_correlations_for_groups(observation_groups,
                                                                       features_to_analyse,
                                                                       ee_reducer)

            with st.spinner('Plotting Correlation ...'):
                st.session_state.correlation_fig = plot_3d_correlation_scatter_with_heatmap(
                    correlation, features_to_analyse,
                    f"{reducer}'s Correlation 3D Scatter Plot",
                    f"{reducer}'s", observation_groups)

    if 'correlation_fig' in st.session_state:
        with stylable_container(key='correlation_container', css_styles=border_container):
            st.write("###### Global Correlation Analysis")

            st.plotly_chart(st.session_state.correlation_fig)

    #### Importance  ####
    with col4.form('importance_form'):
        st.write("##### Feature Importance")
        features = st.multiselect('Select Input Features',
                                  st.session_state.all_features,
                                  default=st.session_state.selected_features,
                                  label_visibility='collapsed')
        all_importance_proxies = ['Impurity Reduction', 'Permutation Accuracy', 'Shapley']
        importance_proxies = st.multiselect('Select Importance Proxy',
                                            all_importance_proxies,
                                            default=all_importance_proxies)

        use_low_card_col = st.toggle("Low cardinality random feature", value=True)
        use_high_card_col = st.toggle("High cardinality random feature", value=True)

        submit = st.form_submit_button(label='Submit', use_container_width=True, type='primary')

    if submit:
        folds = st.session_state.logo_folds
        all_features = st.session_state.all_features

        impurity_importances, list_shap_values, list_test_sets, feature_names, permutation_importances = (
            get_feature_importance(folds, importance_proxies,
                                   use_high_card_col, use_low_card_col,
                                   all_features))

'''        plot_impurity_importances(impurity_importances)
        plot_permutation_importances(permutation_importances)
        plot_shap_values(folds, list_shap_values, feature_names)'''
