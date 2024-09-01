# Standard Library Imports

# Third-Party Library Imports
import ee
import numpy as np

# Local/Application-Specific Imports


def compute_pairwise_correlations_for_groups(groups, features, reducer):
    """
    Computes the pairwise correlations for the specified features across all groups.

    Args:
        groups (list): List of group dictionaries.
        features (list): List of feature names to compute correlations.
        _reducer (ee.Reducer): The reducer to use for computing correlations.

    Returns:
        dict: A dictionary with group labels as keys and lists of correlation data as values.
    """
    correlations = {group['label']: [] for group in groups}

    for group in groups:
        for i, f1 in enumerate(features):
            for j, f2 in enumerate(features):
                if i < j:
                    correlation = group['feature_image'].select([f1, f2]).reduceRegion(
                        reducer=reducer,
                        geometry=group['aoi_ee'],
                        scale=30,
                        maxPixels=1e9
                    ).get('correlation').getInfo()
                    correlations[group['label']].append((f1, f2, correlation))

    return correlations


def create_average_absolute_correlation_matrix(correlations, features):
    """
    Creates an average absolute correlation matrix from all groups.

    Args:
        correlations (dict): Dictionary with group labels as keys and lists of correlation data as values.
        features (list): List of feature names.

    Returns:
        np.ndarray: Average absolute correlation matrix.
    """
    n = len(features)
    avg_abs_matrix = np.zeros((n, n))
    count_matrix = np.zeros((n, n))

    for group_corrs in correlations.values():
        for f1, f2, corr in group_corrs:
            i, j = features.index(f1), features.index(f2)
            avg_abs_matrix[i, j] += abs(corr)
            avg_abs_matrix[j, i] += abs(corr)
            count_matrix[i, j] += 1
            count_matrix[j, i] += 1

    # Avoid division by zero
    count_matrix[count_matrix == 0] = 1
    avg_abs_matrix /= count_matrix

    # Set diagonal to 1
    np.fill_diagonal(avg_abs_matrix, 1)

    return avg_abs_matrix
