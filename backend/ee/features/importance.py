import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import shap
import streamlit as st

def add_random_col(X, use_high_card_col, use_low_card_col):
    X_random = X.copy()

    if use_high_card_col:
        X_random = np.column_stack((X_random, np.random.RandomState(42).randint(0, 1000, X.shape[0])))

    if use_low_card_col:
        X_random = np.column_stack((X_random, np.random.RandomState(42).randint(0, 10, X.shape[0])))

    return X_random


def check_column_names(X_test_random, feature_names):
    # Before assigning feature names
    print("Shape of X_test_random before assigning feature names:", X_test_random.shape)
    print("First few rows of X_test_random (as ndarray):\n", X_test_random[:5])

    # Convert to DataFrame and assign feature names
    X_test_df = pd.DataFrame(X_test_random, columns=feature_names)

    # After assigning feature names
    print("Column names after assigning feature names:", X_test_df.columns.tolist())
    print("First few rows of X_test_df:\n", X_test_df.head())

    return X_test_df


def get_feature_importance(folds, importance_proxies, use_high_card_col, use_low_card_col, all_features):
    impurity_importances = []
    list_shap_values = []
    list_X_tests = []
    permutation_importances = []

    for fold_idx, fold in enumerate(folds):
        X_train, y_train = fold['X_train'], fold['y_train']
        X_test, y_test = fold['X_test'], fold['y_test']

        # Add random columns
        X_train_random = add_random_col(X_train, use_high_card_col, use_low_card_col)
        X_test_random = add_random_col(X_test, use_high_card_col, use_low_card_col)

        # Update feature names
        feature_names = all_features.copy()
        if use_high_card_col:
            feature_names.append('HIGH_CARD_RANDOM')
        if use_low_card_col:
            feature_names.append('LOW_CARD_RANDOM')

        # Check column names before and after assignment
        X_test_df = check_column_names(X_test_random, feature_names)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_random, y_train)

        for importance_proxy in importance_proxies:
            if importance_proxy == 'Impurity Reduction':
                impurity_importance = pd.Series(model.feature_importances_, index=feature_names)
                impurity_importances.append(impurity_importance)

            elif importance_proxy == 'Permutation Accuracy':
                result = permutation_importance(model, X_test_random, y_test, n_repeats=10, random_state=42)
                perm_importance = pd.Series(result.importances_mean, index=feature_names)
                permutation_importances.append(perm_importance)

            elif importance_proxy == 'Shapley':
                if fold_idx == 0:  # Only process the first fold
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test_df)

    for importance_proxy in importance_proxies:
        if importance_proxy == 'Impurity Reduction':
            df = pd.DataFrame(impurity_importances)
            df.boxplot(rot=90)
            plt.title("Impurity Reduction Feature Importances")
            st.pyplot(plt.gcf())
            plt.clf()

        elif importance_proxy == 'Shapley':
            shap.summary_plot(shap_values, X_test_df)

            # Use st.pyplot to display the plot in Streamlit
            st.pyplot(plt.gcf())
            plt.clf()  # Clear the figure after displaying to avoid overlapping plots

        elif importance_proxy == 'Permutation Accuracy':
            df = pd.DataFrame(permutation_importances)
            df.boxplot(rot=90)
            plt.title("Permutation Feature Importances")
            st.pyplot(plt.gcf())
            plt.clf()

    return impurity_importances, list_shap_values, feature_names, permutation_importances