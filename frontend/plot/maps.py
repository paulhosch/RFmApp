from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geemap import cartoee
import streamlit as st

viridis = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']


def plot_all_aois(aoi_ground_truth_pairs):
    figures = []
    all_centroids = []
    colors = list(viridis)  # Distinct colors for each AOI

    for i, (aoi, ground_truth, label, date) in enumerate(aoi_ground_truth_pairs):
        fig = plt.figure(figsize=(6, 6))

        centroid = aoi.geometry.centroid.iloc[0]
        all_centroids.append((centroid.x, centroid.y))

        bounds = aoi.total_bounds
        min_lon, min_lat, max_lon, max_lat = bounds[0], bounds[1], bounds[2], bounds[3]

        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat

        target_ticks = 3
        lon_interval = lon_range / target_ticks
        lat_interval = lat_range / target_ticks

        lon_interval = round(lon_interval / 10) * 10 if lon_interval > 1 else round(lon_interval, 2)
        lat_interval = round(lat_interval / 10) * 10 if lat_interval > 1 else round(lat_interval, 2)

        ax = fig.add_subplot(projection=ccrs.PlateCarree())
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')

        ground_truth.plot(ax=ax, color='#01549f', transform=ccrs.PlateCarree(), label='Flooded Area')

        # Use a distinct color for each AOI
        aoi_color = '#fde725'  #colors[i % len(colors)]
        aoi.plot(ax=ax, edgecolor=aoi_color, linewidth=2, facecolor='none', transform=ccrs.PlateCarree(),
                 label='Area of Interest')

        cartoee.pad_view(ax, factor=[0.2, 0.2])
        cartoee.add_gridlines(ax, interval=[lon_interval, lat_interval], linestyle='--')

        metric_distance = (lon_range * 111) / 5
        metric_distance = max(1, round(metric_distance))

        cartoee.add_north_arrow(ax, xy=(0.9, 0.15), text_color="black", arrow_color="black", fontsize=20,
                                arrow_length=0.1)
        cartoee.add_scale_bar_lite(ax, length=metric_distance, xy=(0.15, 0.05), fontsize=12, color="black", unit="km")

        plt.tight_layout()
        figures.append((fig, f"***Figure {i + 1}:** {label} - {date.strftime('%Y-%m-%d')}*"))

    # Create marker map
    marker_fig = plt.figure(figsize=(6, 6))
    ax_all = marker_fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax_all.set_facecolor('none')
    marker_fig.patch.set_facecolor('none')

    # Calculate the bounding box for all centroids
    min_lon = min(c[0] for c in all_centroids)
    max_lon = max(c[0] for c in all_centroids)
    min_lat = min(c[1] for c in all_centroids)
    max_lat = max(c[1] for c in all_centroids)

    # Add some padding
    padding = 2  # degrees
    ax_all.set_extent([min_lon - padding, max_lon + padding, min_lat - padding, max_lat + padding])

    ax_all.add_feature(cfeature.COASTLINE)
    ax_all.add_feature(cfeature.BORDERS)
    #ax_all.add_feature(cfeature.OCEAN, facecolor='lightblue')
    #ax_all.add_feature(cfeature.LAND, facecolor='lightgreen')

    for i, (lon, lat) in enumerate(all_centroids):
        ax_all.plot(lon, lat, marker='*', color=colors[i % len(colors)], markersize=15, transform=ccrs.Geodetic(),
                    label=f"{aoi_ground_truth_pairs[i][2]} Site")

    ax_all.legend(frameon=False)
    plt.tight_layout()
    figures.append((marker_fig, "***Figure {0}:** All Site Centroids*".format(len(aoi_ground_truth_pairs) + 1)))

    return figures


def calculate_gridline_interval(bounds, target_ticks):
    if isinstance(bounds[0], (list, tuple)):  # For bounds from ee.Image
        min_lon, min_lat = bounds[0]
        max_lon, max_lat = bounds[2]
    else:  # For bounds from GeoDataFrame
        min_lon, min_lat, max_lon, max_lat = bounds

    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat

    # Calculate the larger range to ensure equal spacing
    max_range = max(lon_range, lat_range)

    # Calculate interval aiming for around target_ticks gridlines
    interval = max_range / target_ticks

    # Round interval to one decimal place
    interval = round(interval, 1)

    # Ensure the interval is not too small
    if interval < 0.1:
        interval = 0.1

    return interval


def cartoee_default_map_customization(fig, ax, bounds):
    # Add Padding inside the Map
    cartoee.pad_view(ax, factor=[0.15, 0.15])

    # Calculate grid line interval
    interval = calculate_gridline_interval(bounds, 2)

    # Add gridlines with calculated interval
    cartoee.add_gridlines(ax, interval=interval, linestyle='--')

    # Calculate the scale bar length based on image bounds
    if isinstance(bounds[0], (list, tuple)):  # For bounds from ee.Image
        lon_range = bounds[2][0] - bounds[0][0]
    else:  # For bounds from GeoDataFrame
        lon_range = bounds[2] - bounds[0]
    metric_distance = max(1, round((lon_range * 111) / 5))


    # Add scale bar
    cartoee.add_scale_bar_lite(
        ax, length=metric_distance, xy=(0.15, 0.05), fontsize=10, color="black", unit="km")

    # Add north arrow
    cartoee.add_north_arrow(ax, xy=(0.9, 0.2), text='', arrow_color="black", arrow_length=0.1)

    # Make Background transparent
    fig.patch.set_facecolor('none')  # Set figure background to transparent
    ax.set_facecolor('none')  # Set axis background to transparent

    return fig,ax

@st.cache_resource
def plot_ee_image(_image, _vis_params, label, _discrete=False):
    # Create the plot
    fig = plt.figure(figsize=(7, 5))

    # Add the image
    ax = cartoee.get_map(_image, vis_params=_vis_params, proj=ccrs.PlateCarree())

    # Add colorbar at the bottom center
    cax = ax.figure.add_axes([0.25, -0.05, 0.5, 0.02])
    cartoee.add_colorbar(ax, cax=cax, vis_params=_vis_params, discrete=_discrete, orientation='horizontal',
                         label=label)

    # Get the bounding box of the image
    bounds = _image.geometry().bounds().getInfo()['coordinates'][0]

    # Add coordinate grid, scale bar, north arrow and make background transparent
    fig,ax = cartoee_default_map_customization(fig,ax,bounds)

    return fig


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
