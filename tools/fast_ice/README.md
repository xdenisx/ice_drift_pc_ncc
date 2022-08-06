# Fast ice deliniation

> #### Basic functionality:
>
> - Deliniate fast ice zones based on ice displacement thresholds
> - Export to Geojson file

## Usage

```python

# Definr paths to drift data in npz format and geotiff file
npz_path = '/home/denis/git/dev/ice_drift_pc_ncc/res_20220703_18_24/pyr/vec/50px_ICEDRIFT_20200328T031456_20200330T025831.npz'
tiff_path = '/data/rrs/s1/kara/2020_March_pairs/003/UPS_XX_S1A_EW_GRDM_1SDH_20200330T025831_20200330T025931_031899_03AEA9_540C.tiff'

# Get object 
f = get_fast_ice(npz_path, tiff_path)

# Plot contours
f.plot_contours()

# Export to Geojson file
f.export_json('/data/rrs/seaice/aux_data/pathfinder_ice_motion/data/test')

```

Code: Denis Demchev, Anders Hildeman

## License
[MIT](https://choosealicense.com/licenses/mit/)