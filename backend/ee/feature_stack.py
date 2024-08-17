import ee
import streamlit as st
from backend.obs_group import *

@st.cache_data()
def add_feature_image_to_group(_i, _group, group_hash):
    date = _group.get('date', 'Unknown date')
    label = _group['label']
    aoi_ee = _group['aoi_ee']
    start_date_ee = _group['start_date_ee']
    end_date_ee = _group['end_date_ee']
    all_features = st.session_state.all_features

    # Get Sentinel-1 data
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(aoi_ee) \
        .filterDate(start_date_ee, end_date_ee) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

    # Check coverage
    count = s1.size().getInfo()
    if count == 0:
        st.error(f'No Sentinel-1 imagery available for {label} ({date}).', icon="⚠")
        st.stop()

    # Create Sentinel-1 mosaic
    s1 = s1.mosaic().clip(aoi_ee)

    # Check if imagery fully covers the AOI
    mosaic_bbox = s1.geometry()
    aoi_contained = mosaic_bbox.contains(aoi_ee).getInfo()

    if aoi_contained:
        st.success(f'Sentinel-1 imagery for {label} ({date}) fully covers the AOI.')
    else:
        st.warning(f'Sentinel-1 imagery for {label} ({date}) does not fully cover the AOI.', icon="⚠")

    # Get DEM
    dem = ee.Image('MERIT/DEM/v1_0_3').select('dem').clip(aoi_ee).rename('DEM')

    feature_image = ee.Image()

    # Define all possible bands and their corresponding images
    VV = s1.select('VV')
    VH = s1.select('VH')
    VV2 = VV.pow(2)
    VH2 = VH.pow(2)

    terrain = ee.Algorithms.Terrain(dem)

    feature_calculation = {
        'VV': VV,
        'VH': VH,
        'angle': s1.select('angle'),
        'VV2_VH2': VV2.multiply(VH2),
        'VV2_plus_VH2': VV2.add(VH2),
        'VV_plus_VH': VV.add(VH),
        'DEM': dem,
        'slope': terrain.select('slope'),
        'aspect': terrain.select('aspect')
    }

    # Add requested bands
    for feature, image in feature_calculation.items():
        if feature in feature_calculation:
            feature_image = feature_image.addBands(image.rename(feature))

    # Remove 'constant' band if it exists
    feature_image = feature_image.select(all_features).clip(aoi_ee)

    # Update group
    update_observation_group(_i, feature_image=feature_image)

    return
