#!/bin/bash

find $1 -name '*.json' \
  | grep 'pyr/vec'  \
  | grep -v 'filter' \
  | xargs -I {} python ~/ice_drift_pc_ncc/tools/alpha_shapes/filter_and_anotate_json.py {}