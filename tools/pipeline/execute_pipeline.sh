#!/bin/bash
#
# Bash script for executing the L-C image alignment pipeline



# ---------------------------------------
# ------------ Start snippet ------------
# ---------------------------------------

# Store current directory
CUR_DIR=$PWD




# ---------------------------------------------
# ------------ Generate pair ------------------
# ---------------------------------------------

# Define root path of ALOS geotiff images
ALOS_GEOTIFF_PATH=${CUR_DIR}/geotiff/ALOS2
# Define root path of s1 geotiff images
S1_GEOTIFF_PATH=${CUR_DIR}/geotiff/S1
# Define root path of image pairs
PAIRS_PATH=${CUR_DIR}/pairs

# Define collocation parameters
COLLOCATE_MAXIMUM_TIMEDELTA=0.5
COLLOCATE_MINIMUM_TIMEDELTA=0.0
COLLOCATE_INTERSECTION_RATIO_THRESHOLD=0.1
MAXIMUM_DRIFT_SPEED=0.4
RESOLUTION=100

# Define length of time windows [hours] in which only one image is picked
TIME_WINDOW=4

# Create pairs
cd /data/rrs/seaice/esa_rosel/code/ice_drift_pc_ncc/tools/geotiff_collocation
python3 collocate.py $ALOS_GEOTIFF_PATH $PAIRS_PATH $COLLOCATE_MAXIMUM_TIMEDELTA $S1_GEOTIFF_PATH $COLLOCATE_MINIMUM_TIMEDELTA $COLLOCATE_INTERSECTION_RATIO_THRESHOLD $MAXIMUM_DRIFT_SPEED
# Cull pairs
python3 cull_pairs.py $PAIRS_PATH $RESOLUTION $TIME_WINDOW 1
# produce index of pairs
python3 produce_index.py $PAIRS_PATH


# ---------------------------------------------
# ------------ Compute drift ------------------
# ---------------------------------------------

# Define root path of drift results
DRIFT_RESULTS_PATH=${CUR_DIR}/drift_results

# Define drift retrieval parameters
DRIFT_GRID_STEP=50
# Define if drift retrieval should be based on hv polarization as well (if hv is present)
INCLUDE_HV_IF_PRESENT=0

# Compute drift results
cd /data/rrs/seaice/esa_rosel/code/drift_algorithm/the_core_package
python3 run_drift_alg.py $DRIFT_GRID_STEP $PAIRS_PATH $DRIFT_RESULTS_PATH $MAXIMUM_DRIFT_SPEED $INCLUDE_HV_IF_PRESENT


# ---------------------------------------------
# ------------ Post-process drift -------------
# ---------------------------------------------


# Post-process drift results
cd /data/rrs/seaice/esa_rosel/code/drift_algorithm/tools
python3 drift_post_proc.py $DRIFT_RESULTS_PATH $PAIRS_PATH $DRIFT_GRID_STEP $DRIFT_GRID_STEP


# ---------------------------------------
# ------------ Align images -------------
# ---------------------------------------

# Define root path of alignments
ALIGNMENTS_PATH=${CUR_DIR}/alignments

# Define alignment method
ALIGNMENT_METHOD=piecewise-affine
ALIGNMENT_POLYNOMIAL_ORDER=3

# Align images
cd /data/rrs/seaice/esa_rosel/code/ice_drift_pc_ncc/tools/align
python3 align.py $PAIRS_PATH $DRIFT_RESULTS_PATH $ALIGNMENTS_PATH $ALIGNMENT_METHOD $ALIGNMENT_POLYNOMIAL_ORDER 




# ---------------------------------------
# ------------ End snippet --------------
# ---------------------------------------



# Navigate home again
cd $CUR_DIR




