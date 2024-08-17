from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from geemap import cartoee
import streamlit as st


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
    # Create the chart
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