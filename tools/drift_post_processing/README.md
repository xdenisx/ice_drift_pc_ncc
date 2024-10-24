# Ice drift post processing tools

> #### Basic functionality of the package:
>
> - Operates with output from different ice drift algorithms (currently supports Chalmers ice drift algorithm output)
> - Georeferencing of ice displacement vectors
> - Outliers filtering based on local homogenity criteria (direction and length, for details see Demchev et. al 2017)
> - Ice deformation calculation
> - Apllying land mask
> - Export to vector (geojson/ESRI shapefile) and raster GeoTIFF formats

## Usage

```python
%cd ~/git/ice_drift_pc_ncc/tools/drift_post_processing

from drift_post_proc import *

# Open a file containing ice drift data along with corresponding geotiff files. A path to land mask should be defined in the land_mask_path parameter.
d = driftField('/data/rrs/seaice/esa_rosel/L_C_pairs/north_greenland/wb_dd/drift_results/010/mat/CTU_drift_20191209T174024-20191210T180623.mat',
               '/data/rrs/seaice/esa_rosel/L_C_pairs/north_greenland/wb_dd/pairs/010/UPS_XX_ALOS2_XX_XXXX_XXXX_20191209T174024_20191209T174116_0000326248_001001_ALOS2299501900-191209.tiff',
               '/data/rrs/seaice/esa_rosel/L_C_pairs/north_greenland/wb_dd/pairs/010/UPS_XX_S1B_EW_GRDM_1SDH_20191210T180623_20191210T180727_019306_024746_7A10.tiff', land_mask_path='/home/denis/git/ice_drift_pc_ncc/data/ne_50m_land.shp')

# Export ice drift vectors to geojson file. Before the export, outliers filtering is applied.
d.export_vector(out_path='/data/rrs/seaice/esa_rosel/temp', filtered=True, land_mask=True)

# Calculate and export ice deformation (divergence, shear, curl and total deformation) to the defined directory. Land mask is applied.
d.calc_defo(out_path='/data/rrs/seaice/esa_rosel/temp', land_mask=True)
```

Code: Denis Demchev, Anders Hildeman

## License
[MIT](https://choosealicense.com/licenses/mit/)