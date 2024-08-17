import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import shap


def add_random_col(X, use_high_card_col, use_low_card_col):
    X_random = X.copy()

    if use_high_card_col:
        X_random = np.column_stack((X_random, np.random.RandomState(42).randint(0, 1000, X.shape[0])))

    if use_low_card_col:
        X_random = np.column_stack((X_random, np.random.RandomState(42).randint(0, 10, X.shape[0])))

    return X_random

def get_feature_importance(folds, importance_proxies, use_high_card_col, use_low_card_col, all_features):
    impurity_importances = []
    shap_values_all = []
    list_shap_values = list()
    list_test_sets = list()
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

        # Ensure that the feature names match the columns in X_test_random
        if len(feature_names) != X_test_random.shape[1]:
            raise ValueError(f"Length mismatch: Expected {X_test_random.shape[1]} feature names, "
                             f"but got {len(feature_names)}.")

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_random, y_train)

        for importance_proxy in importance_proxies:
            if importance_proxy == 'Impurity Reduction':
                impurity_importance = pd.Series(model.feature_importances_, index=feature_names)
                impurity_importances.append(impurity_importance)

            elif importance_proxy == 'Shapley':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train_random)
                shap_values_all.append(shap_values)

                list_shap_values.append(shap_values)
                list_test_sets.append(fold_idx)

            elif importance_proxy == 'Permutation Accuracy':
                result = permutation_importance(model, X_test_random, y_test, n_repeats=10, random_state=42)
                perm_importance = pd.Series(result.importances_mean, index=feature_names)
                permutation_importances.append(perm_importance)

    return impurity_importances, list_shap_values, list_test_sets, feature_names, permutation_importances
