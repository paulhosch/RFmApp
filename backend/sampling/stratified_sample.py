import pandas as pd

def sample_points_in_polygon(poly, n, label):
    sample_gs = poly.sample_points(n).explode()
    sample_df = sample_gs.get_coordinates(ignore_index=True)
    sample_df['class'] = label
    sample_df.columns = ['lon', 'lat', 'class']
    return sample_df

def get_stratified_sample(aoi, ground_truth, flooded_size, non_flooded_size):
    flooded_poly = ground_truth.geometry
    non_flooded_poly = aoi.geometry.iloc[0].difference(flooded_poly)
    flooded_sample = sample_points_in_polygon(flooded_poly, flooded_size, 1)
    non_flooded_sample = sample_points_in_polygon(non_flooded_poly, non_flooded_size, 0)
    return pd.concat([flooded_sample, non_flooded_sample])
