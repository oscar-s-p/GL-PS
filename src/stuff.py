from astropy.table import QTable, Table
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
from astropy.visualization import simple_norm

import numpy as np
import matplotlib.pyplot as plt
from photutils.psf import extract_stars


def coordsToStars(image, peaks_tbl, cutout_size=25,):
    if isinstance(peaks_tbl, QTable)==False:
        raise Exception("peaks_tbl must be type astropy.table.QTable")
    
    cutout_size = 25 #must be odd
    hsize = (cutout_size - 1) / 2
    x = peaks_tbl['x_peak']  
    y = peaks_tbl['y_peak']  
    mask = ((x > hsize) & (x < (image.shape[1] -1 - hsize)) & (y > hsize) & (y < (image.shape[0] -1 - hsize))) 

    stars_tbl = Table()
    stars_tbl['x'] = x[mask]  
    stars_tbl['y'] = y[mask]

    mean_val, median_val, std_val = sigma_clipped_stats(image, sigma=2)  
    image -= median_val 

    nddata = NDData(data=image)  

    stars = extract_stars(nddata, stars_tbl, size=cutout_size)

    nrows = int(np.ceil(np.sqrt(len(stars.data))))
    ncols = int(np.ceil(np.sqrt(len(stars.data))))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
                           squeeze=True)
    ax = ax.ravel()

    print(str(len(stars.data))+" stars")

    for i in range(0,len(stars)):
        norm = simple_norm(stars[i], 'log', percent=99.0)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
    
    return stars
