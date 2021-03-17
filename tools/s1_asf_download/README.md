# Seninel-1 data downloading from ASF web-server

The two scripts are for downloading of Sentinel-1A/B Level-1 GRD data

The current version operates with GeoJSON files with POINT or POLYGON geometry. To collect metadata for downloading 
from Alaska SAR Facility server you need to create account there. For that, please go:

>  https://urs.earthdata.nasa.gov/users/new


After the registration is complete you can use the scripts. Then you should create files with a geometry covering an area you are interested.
You can use QGIS, ArcMap or other software and store it in GeoJSON format.

1. Metadata gathering script

```
python asf_download.py /PATH/TO/GEO/FILE Sentinel-1B 20210201 20210203 EW HH+HV GRD_MD 
```

The result will be stored in a local folder:

> ./metalinks/Sentinel-1B_202102010000-202102030000.metalink

2. Data downloading script 

```
python asf_meta_download.py /PATH/TO/METALINK/FILE/FROM/STEP/1 /PATH/TO/OUTPUT/FOLDER 
```

The data will be downloaded in '/PATH/TO/OUTPUT/FOLDER' folder