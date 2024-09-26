# Sentinel-1 data downloading from ASF web-server

The two scripts are for downloading of Sentinel-1A/B Level-1 GRD data

The current version operates with GeoJSON files with POINT or POLYGON geometry. To collect metadata for downloading 
from Alaska SAR Facility server you need to create account there. For that, please go:

>  https://urs.earthdata.nasa.gov/users/new


After the registration is complete you can use the scripts. Then you should create files with a geometry covering an area you are interested.
You can use QGIS, ArcMap or other software and store it in GeoJSON format.

1. Geometry file creation

Let's create a geojson file with POINT geometry over the Fram Strait
```
%run make_geo_file.py point 0,79 /home/denis/git/dev/ice_drift_pc_ncc/tools/s1_asf_download/geo_files/test_fs.json
```

2. Metadata gathering script

Generic example:

```
python asf_meta_download.py /PATH/TO/OUTPUT/DIRECTORY /PATH/TO/GEO/FILE /PLATFORM/ /DATE/ /TIME/ /MINIMUM/HOURS/APART/ /MAXIMUM/HOURS/APART/ /MODE/ /GRD_LEVEL/ /POLARIZATION/ 
```

Let's download data over the Fram Strait (0, 79N) for January 1st of 2020 over 12 hours from 00UTC:

```
python asf_meta_download.py . geo_files/test_fs.json Sentinel-1A,Sentinel-1B 20200101 000000 0 12 EW GRD_MD HH+HV  
```

The result will be stored in a local folder:

> ./metalinks/Sentinel-1B_202102010000-202102030000.metalink

2. Data downloading script 

Generic example:

```
python download_all.py /PATH/TO/METALINK/FILE/FROM/STEP/1 /PATH/TO/OUTPUT/FOLDER 
```

Download data from metadata we formed in Step 1:

```
python download_all.py ./metalinks/Sentinel-1A,Sentinel-1B_202001010000-202001011200.metalink /data/rrs/seaice/aux_data/s1/test
```

The data will be downloaded in '/PATH/TO/OUTPUT/FOLDER' folder
