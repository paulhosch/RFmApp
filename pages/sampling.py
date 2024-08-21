import streamlit as st
from streamlit_extras.chart_container import chart_container
from streamlit_extras.stylable_container import stylable_container
from streamlit_option_menu import option_menu
from sklearn.model_selection import StratifiedKFold

from backend.obs_group import *
from backend.sampling import unify_ground_truth, get_stratified_sample, get_group_X_y, get_logo_folds
from frontend.map import plot_sample_coordinates
from frontend.chart import plot_kfold_splits
from static.styles import opt_menu_style, training_container, testing_container
def sampling():
    if are_observation_groups_valid_and_covered():
        pass
    else:
        st.error("Observation groups are not properly defined.", icon="⚠")

    observation_groups = get_all_observation_groups()

    col1, col2 = st.columns([1, 5])

    with col1:
        with st.form("sampling_design"):
            st.write("#### Sampling Design")
            mode_input_dummy = st.selectbox("**Sampling Mode**", ['Grouped Stratified Random Sampling'])
            total_size = st.number_input("**Samples per Group**", min_value=100, value=500,
                                         step=10)
            allocation = st.selectbox("**Class Allocation**", ["Proportional", "Equalized"])
            st.write("#### Validation Design")

            outer_input_dummy = st.selectbox("**Outer Validation Loop**", ['LOGO-CV'])

            inner_input_dummy = st.selectbox("**Inner Validation Loop**", ['SKF-CV'])
            k_folds = st.number_input("**K-Folds**", min_value=2, value=5, step=1)
            submitted = st.form_submit_button("Create Samples", use_container_width=True, type="primary")

        if submitted:
            progress_bar = st.progress(0)
            total_groups = len(observation_groups)
            for i, group in enumerate(observation_groups):
                progress = (i + 1) / total_groups
                progress_bar.progress(progress, text=f"Sampling from Group {group['label']}")

                group_hash = hash_single_observation_group(group)
                aoi = group['aoi']
                ground_truth = unify_ground_truth(group)

                with st.spinner(text=f"Getting sample points from {group['label']}"):
                    if allocation == "Equalized":
                        flooded_size = non_flooded_size = total_size // 2
                    else:  # Proportional
                        flooded_ratio = ground_truth.geometry.area.iloc[0] / aoi.geometry.area.iloc[0]
                        flooded_size = int(total_size * flooded_ratio)
                        non_flooded_size = total_size - flooded_size

                    sample_coordinates = get_stratified_sample(aoi, ground_truth, flooded_size, non_flooded_size)

                with st.spinner(text=f"Compiling features and labels for observation group {group['label']}"):
                    X, y = get_group_X_y(group, group_hash, sample_coordinates)

                    update_observation_group(i, sample_coordinates=sample_coordinates, X=X, y=y)


            progress_bar.empty()

            with st.spinner(text="Creating LOGO Folds ..."):
                st.session_state.logo_folds = get_logo_folds(observation_groups)
                # Stratified K-Fold CV
                skf = StratifiedKFold(n_splits=k_folds)
                st.session_state.skf = skf

            st.success("Sampling and Splitting completed successfully!")
            st.rerun()

    with col2: #
        if 'logo_folds' in st.session_state:
            folds = st.session_state.logo_folds
            st.write("### Nested Cross-Validation")
            cols = st.columns([2, 1])
            with cols[0]:
                st.write("#### Outside Loop: Leave-One-Group-Out (LOGO)")
            with cols[1]:
                st.write("#### Inside Loop: Stratified K-Fold Cross Validation (SKF-CV)")

            fold_names = [f"Fold {i + 1}" for i in range(len(folds))]
            with cols[0]:
                selected_fold = option_menu(
                    menu_title=None,
                    options=fold_names,
                    orientation="horizontal",
                    styles=opt_menu_style,
                )

            selected_fold_idx = fold_names.index(selected_fold)
            fold = folds[selected_fold_idx]

            # Layout for each fold: first column for testing group, other columns for training groups
            cols = st.columns(len(fold['train_groups']) + 1)

            # Display the testing group in the first column
            test_group_idx = fold['test_group']
            test_group = observation_groups[test_group_idx]

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                with stylable_container(key=f"test-container_{selected_fold_idx}",
                                        css_styles=testing_container):
                    st.write('##### Testing Split ')
                    st.write(f"**{test_group['label']}**")
                    if 'sample_coordinates' in test_group:
                        sample = test_group['sample_coordinates']

                        # Merge sample coordinates with feature data
                        sample_with_features = sample.join(test_group['X'].reset_index(drop=True))

                        flooded_count = sample[sample['class'] == 1].shape[0]
                        non_flooded_count = sample[sample['class'] == 0].shape[0]
                        with chart_container(sample_with_features):
                            fig = plot_sample_coordinates(sample, test_group['aoi'])
                            st.pyplot(fig)
                        st.write(
                            f"***Figure {test_group['label']}:** {test_group['label']} Samples: **{flooded_count} "
                            f"Flooded** Samples in Blue and **{non_flooded_count} "
                            f"Non-Flooded** Samples in Green Within the Area of Interest in Red*")
                    else:
                        st.write(f"No samples generated for {test_group['label']} yet.")

            with col2:
                with stylable_container(key=f"train-container-{selected_fold_idx}",
                                        css_styles=training_container):
                    st.write('##### Training Split')
                    for col_idx, train_group_idx in enumerate(fold['train_groups']):
                        train_group = observation_groups[train_group_idx]

                        st.write(f"**{train_group['label']}**")
                        if 'sample_coordinates' in train_group:
                            sample = train_group['sample_coordinates']
                            # Merge sample coordinates with feature data
                            sample_with_features = sample.join(test_group['X'].reset_index(drop=True))

                            flooded_count = sample[sample['class'] == 1].shape[0]
                            non_flooded_count = sample[sample['class'] == 0].shape[0]
                            with chart_container(sample_with_features):
                                fig = plot_sample_coordinates(sample, train_group['aoi'])
                                st.pyplot(fig)
                            st.write(
                                f"***Figure {train_group['label']}:** {train_group['label']} Samples: **{flooded_count}"
                                f" Flooded** Samples in Blue and **{non_flooded_count}"
                                f" Non-Flooded** Samples in Green Within the Area of Interest in Red*")
                        else:
                            st.write(f"No samples generated for {train_group['label']} yet.")

            with col3:
                with stylable_container(key=f"train-container-{selected_fold_idx}-kfold",
                                        css_styles=training_container):
                    st.write('##### K-Folds of the Training Split ')

                    skf = st.session_state.skf

                    df, fig = plot_kfold_splits(fold['X_train'], fold['y_train'], skf, 5)
                    with chart_container(df):
                        st.pyplot(fig)