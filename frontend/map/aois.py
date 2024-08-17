from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geemap import cartoee
import streamlit as st

from static.vis_params import viridis

def plot_all_aois(aoi_ground_truth_pairs):
    figures = []
    all_centroids = []
    colors = list(viridis)  # Distinct vis_params for each AOI

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
        aoi_color = '#fde725'  #vis_params[i % len(vis_params)]
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