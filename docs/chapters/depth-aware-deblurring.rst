The details of the algorithm from Xu and Jia :cite:`XU2012` and its implementation will be described here...

First-Pass Estimation
+++++++++++++++++++++

Disparity Estimation
--------------------

Disparity Map
,,,,,,,,,,,,,

- :red:`Find disparity map of a blurred stereo image pair.`
- :red:`down-sampling for blur reducing`
- :red:`different stereo algorithm as in paper. This shouldn't effect overall result`
- :red:`violation of stereo matching condition. handle boundary pixel separately`


Cross-Checking
''''''''''''''
- :red:`...`


Region-Tree Construction
------------------------


PSF Estimation for Top-Level Regions
------------------------------------


PSF Propagation
---------------


Blur Removal
------------


Second-Pass Estimation
++++++++++++++++++++++

Disparity Update
----------------


PSF Estimation
--------------