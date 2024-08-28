from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geemap import cartoee
import streamlit as st
from static.vis_params import viridis
import numpy as np
import matplotlib.path as mpath
from matplotlib.patches import ConnectionPatch
import hydralit_components as hc

def plot_all_aois(aoi_ground_truth_pairs):
    figures = []
    all_bounds = []
    colors = list(viridis)  # Distinct vis_params for each AOI

    for i, (aoi, ground_truth, label, date) in enumerate(aoi_ground_truth_pairs):
        fig = plt.figure(figsize=(6, 6))

        bounds = aoi.total_bounds
        all_bounds.append(bounds)
        min_lon, min_lat, max_lon, max_lat = bounds

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

        aoi_color = '#fde725'
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

        # Create AOI map
    aoi_fig = plt.figure(figsize=(10, 8))  # Increased figure size to accommodate inset
    ax_all = aoi_fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax_all.set_facecolor('none')
    aoi_fig.patch.set_facecolor('none')

    # Calculate the bounding box for all AOIs
    min_lon = min(bounds[0] for bounds in all_bounds)
    max_lon = max(bounds[2] for bounds in all_bounds)
    min_lat = min(bounds[1] for bounds in all_bounds)
    max_lat = max(bounds[3] for bounds in all_bounds)

    padding_left = 2  # degrees
    padding_right = 6  # degrees
    padding_bottom = 2  # degrees
    padding_top = 6  # degrees
    ax_all.set_extent([
        min_lon - padding_left,
        max_lon + padding_right,
        min_lat - padding_bottom,
        max_lat + padding_top
    ])

    ax_all.add_feature(cfeature.OCEAN, facecolor='#01549f')
    ax_all.add_feature(cfeature.LAND, facecolor='#c9d2d3')

    for i, (aoi, _, label, _) in enumerate(aoi_ground_truth_pairs):
        color = '#fde725'
        aoi.plot(ax=ax_all, edgecolor=color, linewidth=2, facecolor='none', transform=ccrs.PlateCarree(),
                 label=f"{label} Site")

        # Add label to the right side of each AOI
        bounds = aoi.total_bounds
        label_x = bounds[2] + 0.4  # Use the maximum longitude (right side of the AOI)
        label_y = (bounds[1] + bounds[3]) / 2  # Use the middle latitude

        ax_all.text(label_x, label_y, label, fontsize=12, ha='left', va='center',
                    transform=ccrs.PlateCarree(), color=color)

    # Calculate the center of all AOIs
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    # Define inset position and size (adjust these values to move the inset)
    inset_left = 0.475  # Distance from left edge of figure (0 to 1)
    inset_bottom = 0.625  # Distance from bottom edge of figure (0 to 1)
    inset_width = 0.3  # Width of inset (0 to 1)
    inset_height = 0.3  # Height of inset (0 to 1)

    # Create inset axes for global view
    ax_inset = aoi_fig.add_axes([inset_left, inset_bottom, inset_width, inset_height],
                                projection=ccrs.Orthographic(center_lon, center_lat))
    ax_inset.set_global()

    # Add features to inset map
    #ax_all.add_feature(cfeature.COASTLINE)

    ax_inset.add_feature(cfeature.OCEAN, facecolor='#efefee')
    ax_inset.add_feature(cfeature.LAND, facecolor='#c9d2d3')

    # Plot a point on the inset map to show the center of the main map
    ax_inset.plot(center_lon, center_lat, 'ro', markersize=6, transform=ccrs.Geodetic(), color=color)

    # Add a circular boundary to the Orthographic projection
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax_inset.set_boundary(circle, transform=ax_inset.transAxes)



    #plt.tight_layout()
    figures.append(
        (aoi_fig, "***Figure {0}:** All Areas of Interest with Global Inset*".format(len(aoi_ground_truth_pairs) + 1)))

    return figures