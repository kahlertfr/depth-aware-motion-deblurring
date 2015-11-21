The details of the algorithm from Xu and Jia :cite:`XU2012` and its implementation will be
described here...

First-Pass Estimation
+++++++++++++++++++++

Disparity Estimation
--------------------

Disparity Map
'''''''''''''

- :red:`Find disparity map of a blurred stereo image pair.`
- :red:`down-sampling for blur reducing`
- :red:`different stereo algorithm as in paper. This shouldn't effect overall result`
- :red:`violation of stereo matching condition. handle boundary pixel separately`


Occlusions
''''''''''

:red:`Cross-Checking to find occlusion regions.` In this implementation there is no cross checking
because SGBM handles occluded regions already.

Occlusions are filled with smallest neighbor disparity. Assumption: just objects with small
disparity can be occluded.


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