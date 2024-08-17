import streamlit as st 
from matplotlib import pyplot as plt
import matplotlib.colors as plt_colors
import numpy as np
import pandas as pd


@st.cache_resource()
def plot_kfold_splits(X, y, _cv, k):
    """
    Plot K-Fold splits along with class labels using specified colors.

    Parameters:
    X (array-like): Feature data.
    y (array-like): Target labels.
    _cv: Cross-validation object.
    k (int): Number of splits.

    Returns:
    df (pd.DataFrame): DataFrame with split information.
    fig (plt.Figure): Figure object for the plot.
    """
    fig, ax = plt.subplots()

    # Create color maps for training/testing and class labels
    cv_cmap = plt_colors.ListedColormap(['#440154', '#fde725'])
    class_cmap = plt_colors.ListedColormap(['#2ca02c', '#01549f'])

    # Initialize a DataFrame to store split information
    split_info = []

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(_cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Store the split information in the DataFrame
        for idx in range(len(indices)):
            split_info.append({
                'Sample': idx,
                'Fold': ii,
                'Split': 'Validation' if indices[idx] == 1 else 'Training' if indices[idx] == 0 else 'Unassigned',
                'Class': y[idx]
            })

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=10,
            cmap=cv_cmap,
            vmin=-0.2,
            vmax=1.2,
            label='Training' if ii == 0 else 'Validation'
        )
    # Convert the split information to a DataFrame
    df = pd.DataFrame(split_info)

    # Plot the data classes at the end
    scatter = ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=10, cmap=class_cmap
    )

    # Adding legend for the classes
    legend1 = ax.legend(*scatter.legend_elements(), loc='lower left', bbox_to_anchor=(0.5, -0.3), frameon=False, facecolor='none')
    ax.add_artist(legend1)

    # Formatting
    yticklabels = list(range(k)) + ["Class"]
    ax.set(
        yticks=np.arange(k + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample",
        ylabel="SKF-CV Folds",
        ylim=[k + 1.2, -0.2],
    )

    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    # Adding legend for the splits
    ax.legend(['Validation', 'Training'], loc='lower right',frameon=False, bbox_to_anchor=(0.5, -0.3), facecolor='none')

    return df, fig