import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def plot_impurity_importances(impurity_importances):
    df = pd.DataFrame(impurity_importances)
    df.boxplot(rot=90)
    plt.title("Impurity Reduction Feature Importances")
    plt.show()


def plot_permutation_importances(permutation_importances):
    df = pd.DataFrame(permutation_importances)
    df.boxplot(rot=90)
    plt.title("Permutation Feature Importances")
    plt.show()


def plot_shap_values(folds, list_shap_values, feature_names):
    # Concatenate X_test from all folds
    X_test_all = pd.concat([pd.DataFrame(fold['X_test_random']) for fold in folds], axis=0)

    # Check if the number of columns in X_test_all matches the length of feature_names
    if X_test_all.shape[1] != len(feature_names):
        raise ValueError(f"Length mismatch: X_test_all has {X_test_all.shape[1]} columns, "
                         f"but feature_names has {len(feature_names)} elements.")

    # Assign feature names as column names
    X_test_all.columns = feature_names

    # Concatenate SHAP values from all folds
    shap_values_all = np.concatenate([np.array(value) for value in list_shap_values], axis=0)

    # Create and show the SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_all, X_test_all, show=False)
    plt.tight_layout()
    plt.show()

    return plt.gcf()  # Return the figure for potential further use
