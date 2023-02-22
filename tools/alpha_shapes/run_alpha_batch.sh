#!/bin/bash

echo ''
echo 'Making alpha shapes input'

find $1 -name '*.geojson' \
  | xargs -I {} python ~/ice_drift_pc_ncc/tools/alpha_shapes/prepare_alpha_input.py {}

echo ''
echo 'Making alpha shapes'

find $1 -name '*.alpha_proj.txt' \
  | xargs -I {} python ~/ice_drift_pc_ncc/tools/alpha_shapes/run_alpha.py {}

echo ''
echo 'Making shape files'

find $1 -name '*_alpha_shape_*km.txt' \
  | xargs -I {} python ~/ice_drift_pc_ncc/tools/alpha_shapes/alpha_to_shp_contours.py {}