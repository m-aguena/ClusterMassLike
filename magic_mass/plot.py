import astropy.units as u
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import G
from astropy.cosmology import Planck15
from astropy.table import Table, dstack, vstack

import matplotlib.colors as mcolors

def truncate_colormap(cmap_name, min_val=0.0, max_val=1.0, n=100):
    cmap = plt.get_cmap(cmap_name)
    # Extract colors from the specific range (0.0 to 1.0)
    colors = cmap(np.linspace(min_val, max_val, n))
    # Build and return the new truncated colormap
    return mcolors.LinearSegmentedColormap.from_list(f"trunc_{cmap_name}", colors)


sliced_cmap = truncate_colormap(
    "magma", min_val=0.0, max_val=((len(m_bins) - 1) / len(m_bins))
)
