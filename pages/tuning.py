import numpy as np
import time
import json

import streamlit as st
import pandas as pd
import optuna
import optuna.visualization as vis
from optuna.multi_objective import trial

from backend.tuning import evaluate_model, inner_cv_objective, outer_cv_objective
from frontend.chart import display_study_results, plot_confusion_matrix


def tuning():
    if 'logo_folds' not in st.session_state:
        st.error("Sample training and validation Data first.", icon="⚠")

    col1, col2 = st.columns([1,4])

    # Create a form for hyperparameter range inputs
    with col1.form(key="hyperparameter_form"):
        st.write("#### Random Forest Classifier Optimization with Optuna")

        # Slider to select the number of Optuna trials
        n_trials = st.slider("Number of Optuna trials", min_value=5, max_value=1000, value=10, step=1)
        # Select between inner or outer cross-validation
        cv_type = st.selectbox("Select Cross-Validation Loop", ["Inner Cross-Validation", "Outer Cross-Validation"])

        features = st.multiselect('Select Input Features',
                                  st.session_state.all_features,
                                  default=st.session_state.selected_features)

        st.write("#### Hyperparameter Ranges")

        n_estimators_range = st.slider(
            "Number of Trees (n_estimators)",
            min_value=10,
            max_value=10000,
            value=(100, 1000),
            step=10,
            help=(
                "Specifies the number of trees in the forest. "
                "A higher number of trees can improve model performance by reducing variance, "
                "but it also increases computational cost. "
                "Typically, more trees lead to better results, up to a point of diminishing returns."
            )
        )
        criterion_options = st.multiselect(
            "Splitting Criterion (criterion)",
            ["gini", "entropy", "log_loss"],
            default= ["gini", "entropy", "log_loss"],
            help=(
                "Specifies the function to measure the quality of a split in the decision trees. "
                "'gini' uses the Gini impurity, 'entropy' uses information gain, and 'log_loss' "
                "is related to the logistic loss function. Different criteria can lead to "
                "different tree structures, potentially affecting model performance."
            )
        )
        max_depth_range = st.slider(
            "Maximum Tree Depth (max_depth)",
            min_value=1,
            max_value=1000,
            value=(10, 100),
            step=1,
            help=(
                "Limits the maximum depth of each tree in the forest. A smaller value "
                "constrains the model, preventing overfitting, but may cause underfitting. "
                "Higher values allow deeper trees, capturing more complexity but increasing "
                "the risk of overfitting."
            )
        )
        min_samples_split_range = st.slider(
            "Minimum Samples to Split a Node (min_samples_split)",
            min_value=1,
            max_value=32,
            value=(2, 20),
            step=1,
            help=(
                "The minimum number of samples required to split an internal node. "
                "A smaller value allows for more splits, creating a deeper tree, "
                "which can lead to overfitting. Conversely, a larger value restricts "
                "splits, producing a shallower tree that may generalize better but could "
                "also underfit."
            )
        )
        min_samples_leaf_range = st.slider(
            "Minimum Samples at a Leaf Node (min_samples_leaf)",
            min_value=1,
            max_value=50,
            value=(1, 10),
            step=1,
            help=(
                "The minimum number of samples required to be at a leaf node. "
                "This parameter controls the minimum size of the terminal nodes (leaves). "
                "A smaller value allows the model to capture more specific patterns, potentially leading to overfitting. "
                "Higher values create larger leaves, which can help generalize the model but may cause underfitting. "
            )
        )

        max_features_range = st.slider(
            "Number of Features to Consider at Each Split (max_features)",
            min_value=1,
            max_value=len(features),  # Dynamic maximum value based on the number of features
            value=(1, len(features)),  # Default to a practical range from 1 to sqrt(n_features)
            step=1,
            help=(
                "The number of features to consider when looking for the best split. "
                "A lower value (e.g., 1) forces the model to consider fewer features, "
                "which can introduce more randomness and potentially reduce overfitting. "
                "Higher values allow the model to consider more features, up to the total "
                "number of features (n_features), which can improve model performance but "
                "may increase the risk of overfitting. "
            )
        )

        max_samples = st.slider(
            "Maximum Samples to Draw (max_samples)",
            min_value=0.1,
            max_value=1.0,
            value=(0.5, 1.0),  # Default range for float values
            step=0.01,
            help=(
                "Specifies the number of samples to draw from the dataset to train each base estimator "
                "when bootstrap is enabled. "
                "A value of 1.0 means using all samples, while lower values can increase randomness "
                "and may help reduce overfitting. "
            )
        )

        # Submit button inside the form
        submit_button = st.form_submit_button(label="Start Optimization", use_container_width= True, type='primary')

    # Check if the form was submitted
    if submit_button:
        st.empty()
        # Load cross-validation folds from the session state (assumed to be precomputed)
        folds = st.session_state.logo_folds
        inner_cv = st.session_state.skf

        # Initialize the progress bar
        progress_bar = col2.progress(0)
        # Record the start time
        start_time = time.time()

        # Define the callback function to update the progress bar
        def update_progress_bar(study, trial):
            elapsed_time = time.time() - start_time
            avg_time_per_trial = elapsed_time / (trial.number + 1)
            estimated_total_time = avg_time_per_trial * n_trials
            estimated_remaining_time = estimated_total_time - elapsed_time

            progress = (trial.number + 1) / n_trials
            progress_bar.progress(progress, text=f"Trial {trial.number + 1}/{n_trials} | "
                               f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} | "
                               f"Estimated Remaining Time: {time.strftime('%H:%M:%S', time.gmtime(estimated_remaining_time))}")

        # Create an Optuna study to maximize the objective function
        study = optuna.create_study(direction='maximize')

        if cv_type == "Inner Cross-Validation":
            # Use the first fold to optimize hyperparameters
            first_fold = folds[0]
            X_train_first, y_train_first = first_fold['X_train'][features], first_fold['y_train']  # Filter the features

            # Optimize the study with the defined objective function on the first fold
            study.optimize(lambda trial: inner_cv_objective(
                trial,
                X_train_first,
                y_train_first,
                inner_cv,
                n_estimators_range,
                criterion_options,
                max_depth_range,
                min_samples_split_range,
                min_samples_leaf_range,
                max_features_range,
                max_samples
            ), n_trials=n_trials, callbacks=[update_progress_bar])

        elif cv_type == "Outer Cross-Validation":
            # Optimize the study using the outer cross-validation objective function
            study.optimize(lambda trial: outer_cv_objective(
                trial,
                folds,
                features,
                n_estimators_range,
                criterion_options,
                max_depth_range,
                min_samples_split_range,
                min_samples_leaf_range,
                max_features_range,
                max_samples
            ), n_trials=n_trials, callbacks=[update_progress_bar])

        # Store the study in the session state
        st.session_state.optuna_study = study

        progress_bar.empty()

        with col2.spinner('Evaluating best model ...'):
            st.session_state.best_params = study.best_params
            st.session_state.model_evaluation_results = evaluate_model(folds, st.session_state.best_params)

    with col1:
        if 'optuna_study' in st.session_state and 'model_evaluation_results' in st.session_state:
            st.write("**Best Hyperparameters Found**")
            # Convert the best_params dictionary to a DataFrame for a better display
            best_params_df = pd.DataFrame(list(st.session_state.best_params.items()), columns=["Hyperparameter", "Value"])
            st.dataframe(best_params_df, hide_index=True, use_container_width=True)  # Display as a table

            outer_results_best, outer_results_default, all_y_true, all_y_pred_best, all_y_pred_default = st.session_state.model_evaluation_results

            st.write("**Model Performance with Outer CV**")

            for metric in outer_results_best.keys():
                mean_best = np.mean(outer_results_best[metric])
                std_best = np.std(outer_results_best[metric])
                mean_default = np.mean(outer_results_default[metric])
                std_default = np.std(outer_results_default[metric])
                delta = mean_best - mean_default

                st.metric(
                    label=f"Mean {metric.capitalize()}",
                    value=f"{mean_best:.4f} (±{std_best:.4f})",
                    delta=f"{delta:.4f}",
                    help=f"Default: {mean_default:.4f} (±{std_default:.4f})"
                )

            # Plot and display the confusion matrix for the combined results
            fig = plot_confusion_matrix(all_y_true, all_y_pred_best, "Confusion Matrix with Outer CV")
            st.plotly_chart(fig)

    with col2:
        # If a study already exists in the session state, display it
        if 'optuna_study' in st.session_state:
            display_study_results(st.session_state.optuna_study)
            return

