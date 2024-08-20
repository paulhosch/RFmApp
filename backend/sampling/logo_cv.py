
import pandas as pd

def get_logo_folds(observation_groups):
    """
    Generate Leave-One-Group-Out (LOGO) folds.

    Parameters:
    observation_groups (list): List of dictionaries containing 'X' and 'y' DataFrames for each group.

    Returns:
    folds (list): List of dictionaries with training and testing data and group info.
    """
    folds = []
    X_list = [group['X'] for group in observation_groups]
    y_list = [group['y'] for group in observation_groups]

    for i in range(len(observation_groups)):
        X_test, y_test = X_list[i], y_list[i]
        X_train = pd.concat([X for j, X in enumerate(X_list) if j != i], axis=0)
        y_train = pd.concat([y for j, y in enumerate(y_list) if j != i], axis=0)

        # Sort X_train and y_train based on y_train
        sorted_indices = y_train.sort_values(by='label').index
        X_train_sorted = X_train.loc[sorted_indices]
        y_train_sorted = y_train.loc[sorted_indices]

        # Sort X_test and y_test based on y_test
        sorted_indices_test = y_test.sort_values(by='label').index
        X_test_sorted = X_test.loc[sorted_indices_test]
        y_test_sorted = y_test.loc[sorted_indices_test]

        # Store the fold information
        fold_info = {
            'X_train': X_train_sorted,
            'y_train': y_train_sorted,
            'X_test': X_test_sorted,
            'y_test': y_test_sorted,
            'train_groups': [j for j in range(len(observation_groups)) if j != i],
            'test_group': i
        }

        folds.append(fold_info)

    return folds
