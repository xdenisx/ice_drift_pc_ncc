# Donwload ERA5 data

Usage: 

```
python ERA5ForTiff.py /PATH/TO/GEOTIFF/ /PATH/TO/STORE/DATA/ /VARIABLE/ /HOURS/PRIOR/ /HOURS/AFTER/
```


'/PATH/TO/GEOTIFF/' is the path to the geotiffs for which ERA5 data should be acquired.

'/PATH/TO/STORE/DATA/' path to directory to store the NetCDF file of ERA5 data, as well as the generated images. If netCDF file already exists, the script will disregard download and use the existing file instead (this is good if the existing file is what you want, it is bad if the existing file does not represent the data you are interested in).

'/VARIABLE/' The ERA5 variable to use.

'/HOURS/PRIOR/' How many hours before geotiff file should be donwloaded.

'/HOURS/AFTER/' How many hours after geotiff file should be donwloaded.



This script will visualize the ERA5 data compared to the geotiff file.