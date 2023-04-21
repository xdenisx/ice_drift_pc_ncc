# Plot variables of geolocation grid 

Usage: 

```
python plotGeolocationPoints.py /PATH/TO/GEOTIFFS/IMAGE/ /POLARIZATION/ /VARIABLE/
```


'/PATH/TO/GEOTIFFS/IMAGE/' is the path to the geotiffs which geolocation grid points in its metadata.

'POLARIZATION' the text string of indicating the band of the polarization (typically "hh" or "hv"). Running the script without this parameter will give you a list of the available polarizations.

'VARIABLE' name of the variable. For instance "incidenceAngle". Running the script without this parameter will give you a list of the available varaibles.


The script will visualize the spatial data. the script can be run as a module if you want to do further post-processing on the data, see script.