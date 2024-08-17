from shapely import unary_union
import geopandas as gpd


def unify_ground_truth(group):
    """
    Performs a unary union on the geometries of the given GeoDataFrame group and
    returns a new GeoDataFrame containing the resulting multipolygon.

    Parameters:
    group (gpd.GeoDataFrame): GeoDataFrame containing geometries to be unified.

    Returns:
    gpd.GeoDataFrame: A new GeoDataFrame with a single multipolygon.
    """
    ground_truth = group['ground_truth']
    multipolygon = unary_union(ground_truth.geometry)
    ground_truth = gpd.GeoDataFrame(geometry=[multipolygon])

    return ground_truth