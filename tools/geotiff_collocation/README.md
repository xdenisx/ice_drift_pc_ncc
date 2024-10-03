# Geotiff collocation and making a pairs of images within defined time lag

Usage: 

```
python collocate.py /PATH/TO/INPUT/GEOTIFFS /OUTPUT/PATH/FOR/PAIRS TIME_LAG_IN_DAYS (/PATH/TO/INPUT/GEOTIFFS2) (MINIMUM_TIME_LAGS) (INTERSECTION_RATIO_THRESHOLD) (MAXIMUM_DRIFT_SPEED)
```

Here, '()' indicates optional parameter.

'/PATH/TO/INPUT/GEOTIFFS' is the path to the original geotiffs.

'/OUTPUT/PATH/FOR/PAIRS' is the path to place the outputed pairs in.

'TIME_LAG_IN_DAYS' is a floating point number giving the maximum time duration between images in a pair.

'/PATH/TO/INPUT/GEOTIFFS2' is the path to a second place with original geotiffs. 
    If given, every image in '/PATH/TO/INPUT/GEOTIFFS' is attempted to be paired with every image in '/PATH/TO/INPUT/GEOTIFFS2'.
    If not given, the images in '/PATH/TO/INPUT/GEOTIFFS' are instead paired with each other.

'MINIMUM_TIME_LAGS' is a floating point number giving the minimum time duration between images in a pair. 
    If not given, no minimum is used. 

'INTERSECTION_RATIO_THRESHOLD' is a floating point number giving a threshold between the number of overlapping "active "pixels in the paired images and the number of "active" pixels in the image among the pair which has the smallest number of "active" pixels.
    Every image pair with a smaller threshold value than 'INTERSECTION_RATIO_THRESHOLD' will be rejected.
    If not given, a value of 0.34 is used. 

'MAXIMUM_DRIFT_SPEED' is a floating point number giving the fastest speed that ice is allowed/believed to move [in meters per second]. 







# Calculate and show the areas and overlapping area between two images

```
python tiff_areas.py /PATH/TO/FIRST/IMAGE/ /RESOLUTION/ /PATH/TO/SECOND/IMAGE/
```


'/PATH/TO/FIRST/IMAGE/': Path to first image
'/RESOLUTION/': The resolution [in meters per pixel (in each direction)]. For instance, a value of 100 would mean that a pixel corresponds to a 100 x 100 m^2 area.
'/PATH/TO/SECOND/IMAGE/': Path to a second image. If not given, the script computes the (active) area of the first image. If given, the script computes the overlapping (active) area of the two images.



# Find the image pairs with the largest overlaps within each time window. Remove all but the largest overlapping pair in each time window (optional).

```
python cull_pairs.py /PATH/TO/PAIRS/FOLDER/ /RESOLUTION/ /SIZE/OF/TEMPORAL/WINDOW/ /REMOVE/SMALLER/PAIRS/
```


'/PATH/TO/PAIRS/FOLDER/': Path to folder of image pairs
'/RESOLUTION/': The resolution [in meters per pixel (in each direction)]. For instance, a value of 100 would mean that a pixel corresponds to a 100 x 100 m^2 area.
'/SIZE/OF/TEMPORAL/WINDOW/': Size of temporal windows [in hours]
/REMOVE/SMALLER/PAIRS/ : Should the pairs with smaller areas be removed? [1 / 0]








