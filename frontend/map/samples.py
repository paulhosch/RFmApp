from matplotlib import pyplot as plt
import cartopy.crs as ccrs

from .utils import cartoee_default_map_customization

def plot_sample_coordinates(samples_df, aoi):
    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 7), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add AOI
    aoi.plot(ax=ax, edgecolor='red', linewidth=2, facecolor='none', transform=ccrs.PlateCarree(), label='AOI')

    # Plot sampled points
    class_colors = {0: '#2ca02c', 1: '#01549f'}  # Customize colors for each class
    for class_value, color in class_colors.items():
        class_samples = samples_df[samples_df['class'] == class_value]
        ax.scatter(
            class_samples['lon'],
            class_samples['lat'],
            color=color,
            label=f'Class {class_value}',
            transform=ccrs.PlateCarree(),
            marker='x',
            s=50
        )

    # Add legend
    ax.legend(loc='upper right', facecolor='none', frameon=False)

    # Get the bounding box of the AOI
    bounds = aoi.total_bounds

    # Add coordinate grid, scale bar, north arrow and make background transparent
    fig, ax = cartoee_default_map_customization(fig, ax, bounds)

    # Adjust layout to add space between the plots
    plt.subplots_adjust(wspace=0.2)

    return fig
