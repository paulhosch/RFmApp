import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def cmap_to_hex(cmap_name, n_colors=256):
    cmap = plt.get_cmap(cmap_name, n_colors)
    return [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]


features = {
    'VV': {'min': -25, 'max': 5, 'palette':  ['000000', 'FFFFFF']},
    'VH': {'min': -25, 'max': 0, 'palette':  ['000000', 'FFFFFF']},
    'angle': {'palette': cmap_to_hex('viridis')},

    'VV2_VH2': {'palette':  cmap_to_hex('viridis')},
    'VV2_plus_VH2': {'palette':  ['000000', 'FFFFFF']},
    'VV_plus_VH': {'palette':  ['000000', 'FFFFFF']},

    'DEM': {'palette': cmap_to_hex('viridis')},
    'slope': {'palette': cmap_to_hex('viridis')},
    'aspect': {'palette': cmap_to_hex('viridis')}
}