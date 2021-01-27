# ice_drift_pc_ncc
Ice drift retrieval algorithm based on combination of normilized cross-correlation and phase correlation

1. The main script:
cc_bm_parallel_pyr_dev.py

cmd arguments:
"filename1 filename2 block_size search_area grid_step"

Example of usage in IPython: 

```
run cc_bm_parallel_pyr_dev.py ./data/ASA_WSM__20071225T022840_0761_g.tif ./data/ASA_WSM__20071226T115239_7547_g.tif 64 4 100
```

1.1 Config file
cc_config.py

the file contain all algorithm parameters

1.2 The number of CPU's for parallel computing is defined in line 1274
 
1.3 cc_calc_drift.py
Calculate ice drift
 
1.4 cc_calc_defo.py
Calculate ice deformation
 
1.5 cc_calc_drift_filter.py
Erronemous ice drift vectors filtrering by homogenity criteria

1.6
The output of the algorithm is saved in "res_DATE_TIME"
