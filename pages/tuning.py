import streamlit as st
import pandas as pd
import numpy as np
import optuna
import time
from backend.tuning import inner_cv_objective, outer_cv_objective, evaluate_model
from frontend.chart import display_study_results, plot_confusion_matrix

def create_param_input(param_name, default_value, input_type, default_radio="Yes", **kwargs):
    if param_name not in st.session_state:
        st.session_state[param_name] = {"use": default_radio == "Yes", "value": default_value}

    @st.fragment
    def param_fragment():
        with st.container(border=True):
            st.write(f"**{param_name}**")
            col1, col2 = st.columns([1, 3])
            with col1:
                use_param = st.toggle(f"Optimize {param_name}", key=f"{param_name}_toggle",
                                      value=st.session_state[param_name]["use"], label_visibility="collapsed")
            with col2:
                if use_param:
                    if input_type == "slider":
                        value = st.slider(param_name, key=f"{param_name}_slider", label_visibility="collapsed", **kwargs)
                    elif input_type == "multiselect":
                        value = st.multiselect(param_name, key=f"{param_name}_multiselect", label_visibility="collapsed", **kwargs)
                else:
                    st.metric(label=param_name, value=default_value, label_visibility="collapsed")
                    value = default_value

        st.session_state[param_name]["use"] = use_param
        st.session_state[param_name]["value"] = value

    param_fragment()
    return "Yes" if st.session_state[param_name]["use"] else "No", st.session_state[param_name]["value"]

def tuning():
    if 'logo_folds' not in st.session_state:
        st.error("Sample training and validation Data first.", icon="⚠")
        return

    col1, col2 = st.columns([1, 4])

    with col1:
        with st.container(border=True):
            st.write('**Hyperparameter Tuning**')
            n_trials = st.slider("Specify Number of Optimization Trials", min_value=5, max_value=1000, value=10, step=1)
            cv_type = st.selectbox("Select Cross-Validation Loop to get Testing Data", ["Inner Cross-Validation", "Outer Cross-Validation"])

            features = st.multiselect('Select Input Features to use',
                                      st.session_state.all_features,
                                      default=st.session_state.selected_features)

            st.session_state.selected_features = features

            st.write('Select Hyperparameters to optimize')
            with st.container(border=False, height=400):


                use_n_estimators, n_estimators_range = create_param_input("n_estimators", 100, "slider", default_radio="Yes",
                                                                          min_value=10, max_value=10000, value=(100, 1000),
                                                                          step=10)

                use_criterion, criterion_options = create_param_input("criterion", "gini", "multiselect", default_radio="No",
                                                                      options=["gini", "entropy", "log_loss"],
                                                                      default=["gini"])

                use_max_depth, max_depth_range = create_param_input("max_depth", None, "slider", default_radio="Yes",
                                                                    min_value=1, max_value=1000, value=(10, 100), step=1)

                use_min_samples_split, min_samples_split_range = create_param_input("min_samples_split", 2, "slider",
                                                                                    default_radio="No",
                                                                                    min_value=2, max_value=32, value=(2, 20),
                                                                                    step=1)

                use_min_samples_leaf, min_samples_leaf_range = create_param_input("min_samples_leaf", 1, "slider",
                                                                                  default_radio="No",
                                                                                  min_value=1, max_value=50, value=(1, 10),
                                                                                  step=1)

                use_max_features, max_features_range = create_param_input("max_features", 'sqrt', "slider", default_radio="Yes",
                                                                          min_value=1, max_value=len(features),
                                                                          value=(1, len(features)), step=1)

                use_max_samples, max_samples_range = create_param_input("max_samples", None, "slider", default_radio="No",
                                                                        min_value=0.1, max_value=1.0, value=(0.5, 1.0),
                                                                        step=0.01)

        submit_button = st.button("Start Hyperparameter Tuning", type='primary', use_container_width=True)

    if submit_button:
        with col2:
            with st.spinner(""):
                st.empty()
                folds = st.session_state.logo_folds
                inner_cv = st.session_state.skf

                # Compile parameters based on UI input
                param_ranges = {}
                if use_n_estimators == "Yes":
                    param_ranges['n_estimators'] = n_estimators_range
                if use_criterion == "Yes":
                    param_ranges['criterion'] = criterion_options
                if use_max_depth == "Yes":
                    param_ranges['max_depth'] = max_depth_range
                if use_min_samples_split == "Yes":
                    param_ranges['min_samples_split'] = min_samples_split_range
                if use_min_samples_leaf == "Yes":
                    param_ranges['min_samples_leaf'] = min_samples_leaf_range
                if use_max_features == "Yes":
                    param_ranges['max_features'] = max_features_range
                if use_max_samples == "Yes":
                    param_ranges['max_samples'] = max_samples_range


                progress_bar = col2.progress(0)
                start_time = time.time()

                def update_progress_bar(study, trial):
                    elapsed_time = time.time() - start_time
                    avg_time_per_trial = elapsed_time / (trial.number + 1)
                    estimated_total_time = avg_time_per_trial * n_trials
                    estimated_remaining_time = estimated_total_time - elapsed_time

                    progress = (trial.number + 1) / n_trials
                    progress_bar.progress(progress, text=f"Trial {trial.number + 1}/{n_trials} | "
                                                         f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} | "
                                                         f"Estimated Remaining Time: {time.strftime('%H:%M:%S', time.gmtime(estimated_remaining_time))}")

                study = optuna.create_study(direction='maximize')

                if cv_type == "Inner Cross-Validation":
                    first_fold = folds[0]
                    X_train_first, y_train_first = first_fold['X_train'][features], first_fold['y_train']

                    study.optimize(lambda trial: inner_cv_objective(
                        trial, X_train_first, y_train_first, inner_cv, param_ranges
                    ), n_trials=n_trials, callbacks=[update_progress_bar])

                elif cv_type == "Outer Cross-Validation":
                    study.optimize(lambda trial: outer_cv_objective(
                        trial, folds, features, param_ranges
                    ), n_trials=n_trials, callbacks=[update_progress_bar])

                st.session_state.optuna_study = study

                progress_bar.empty()

                with st.spinner('Evaluating best model ...'):
                    st.session_state.best_params = study.best_params
                    st.session_state.model_evaluation_results = evaluate_model(folds, st.session_state.best_params)

    # Display results
    # ... [rest of the code remains the same] ...

    # Display results
    with col2:
        if 'optuna_study' in st.session_state and 'model_evaluation_results' in st.session_state:
            st.write("**Best Hyperparameters Found**")
            best_params_df = pd.DataFrame(list(st.session_state.best_params.items()),
                                          columns=["Hyperparameter", "Value"])
            st.dataframe(best_params_df, hide_index=True, use_container_width=True)

            outer_results_best, outer_results_default, all_y_true, all_y_pred_best, all_y_pred_default = st.session_state.model_evaluation_results

            st.write("**Model Performance with Outer CV**")

            # Create columns for metrics
            metric_cols = st.columns(len(outer_results_best.keys()))

            for i, (metric, col) in enumerate(zip(outer_results_best.keys(), metric_cols)):
                mean_best = np.mean(outer_results_best[metric])
                std_best = np.std(outer_results_best[metric])
                mean_default = np.mean(outer_results_default[metric])
                std_default = np.std(outer_results_default[metric])
                delta = mean_best - mean_default

                col.metric(
                    label=f"Mean {metric.capitalize()}",
                    value=f"{mean_best:.4f}",
                    delta=f"{delta:.4f}",
                    help=f"Best: {mean_best:.4f} (±{std_best:.4f})\nDefault: {mean_default:.4f} (±{std_default:.4f})"
                )

            fig = plot_confusion_matrix(all_y_true, all_y_pred_best, "Confusion Matrix with Outer CV")
            col1.plotly_chart(fig)

    with col2:
        if 'optuna_study' in st.session_state:
            display_study_results(st.session_state.optuna_study)