# SAR image alignment by ice drift compensation

This repository contains a Python class for alignment of a pair of SAR sea ice images by the drift compensation and based on work [Eriksson et al, 2022](https://ieeexplore.ieee.org/abstract/document/9884292).
The code operates with two sequential SAR images in geotiff format and a text file containing ice displacements.
The displacements can be provided either in pixel coordinates (row/column number) or in geographical coordinates (latitude/longitude). In case of using geographical coordinates a parameter ``geocoded`` must be set to ``True``.

<br><br>
An example of the displacement file content (pixel coordinates):
<br><br>
```
# x0, y0, dx, dy
...
1485,3195,-4.0,-71.0
1485,3245,-3.0,-72.0
1485,3295,-2.0,-73.0
...
```

Usage:

```python
from AlignSAR import *

%%time
a = Alignment(img1_path='UPS_XX_ALOS2_XX_XXXX_XXXX_20191028T174022_20191028T174114_0000326209_001001_ALOS2293291900-191028.tiff',
              img2_path='UPS_XX_S1B_EW_GRDM_1SDH_20191031T170040_20191031T170144_018722_02349F_0FA6.tiff',
              displacement_path='fltrd_CTU_drift_20191028T174022-20191031T170040.csv',
              out_path='test_results')
```

<br><br>
An example of the displacement file content (geographical coordinates):
<br>
```
# lon0, lat1, lon2, lat2
...
-3.8136,81.33517,-3.6024,81.3035
-3.6636,81.33665,-3.4552,81.3043
-3.5135,81.33807,-3.3038,81.3046
...
```
  
<br>
Usage (``geocoded=True``):

```python
from AlignSAR import *

%%time
a = Alignment(img1_path='subset_1_of_subset_1_of_S1A_EW_GRDM_1SDH_20221119T072104_20221119T072208_045960_057FE5_E8B1_Orb_Cal_TC_HV.tif',
              img2_path='subset_2_of_S1A_EW_GRDM_1SDH_20221120T080155_20221120T080259_045975.tif',
              displacement_path='subset_19-20_nov_drift_HV.csv',
              out_path='test_res',
              geocoded=True)
```

The code written by Anders Hildeman and Denis Demchev




