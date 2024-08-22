import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap
import streamlit as st

def add_random_col(X, use_high_card_col, use_low_card_col):
    X_random = X.copy()

    if use_high_card_col:
        high_card_random = np.random.RandomState(42).randint(0, 10000, X.shape[0])
        X_random['HIGH_CARD_RANDOM'] = high_card_random

    if use_low_card_col:
        low_card_random = np.random.RandomState(42).randint(0, 10, X.shape[0])
        X_random['LOW_CARD_RANDOM'] = low_card_random

    return X_random
@st.cache_data()
def get_importances(folds, importance_proxies, use_high_card_col, use_low_card_col, features):
    impurity_importances = []
    permutation_importances = []
    SHAP_values_per_fold = []
    X_test_per_fold = []
    feature_names = None

    # Initialize progress bar
    progress_bar = st.progress(0)
    total_folds = len(folds)

    for fold_idx, fold in enumerate(folds):
        progress = (fold_idx + 1) / total_folds
        progress_bar.progress(progress, text=f"Evaluating Feature Importance for Fold {fold_idx}")

        X_train, y_train = fold['X_train'], fold['y_train']
        X_test, y_test = fold['X_test'], fold['y_test']

        # Select only the specified features
        X_train = X_train[features]
        X_test = X_test[features]

        # Add random columns
        X_train_random = add_random_col(X_train, use_high_card_col, use_low_card_col)
        X_test_random = add_random_col(X_test, use_high_card_col, use_low_card_col)

        # Update feature names
        feature_names = X_train_random.columns.tolist()

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_random, y_train['label'])  # Assuming 'label' is the column name for y

        for importance_proxy in importance_proxies:
            if importance_proxy == 'Impurity Reduction':
                impurity_importance = pd.Series(model.feature_importances_, index=feature_names)
                impurity_importances.append(impurity_importance)

            elif importance_proxy == 'Permutation Accuracy':
                result = permutation_importance(model, X_test_random, y_test['label'], n_repeats=50, random_state=42)
                perm_importance = pd.Series(result.importances_mean, index=feature_names)
                permutation_importances.append(perm_importance)

            elif importance_proxy == 'Shapley':
                # Calculate SHAP values for each fold
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_random)
                SHAP_values_per_fold.append(shap_values)
                X_test_per_fold.append(X_test_random)

    progress_bar.empty()
    # Aggregate SHAP values across all folds
    aggregated_shap_values = None
    aggregated_X_test = None
    if 'Shapley' in importance_proxies:
        aggregated_shap_values = np.concatenate(SHAP_values_per_fold)
        aggregated_X_test = pd.concat(X_test_per_fold, axis=0)

    return impurity_importances, permutation_importances, aggregated_shap_values, aggregated_X_test, feature_names