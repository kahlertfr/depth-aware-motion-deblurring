======
Theory
======

Depth-Aware Motion Blur
=======================


Blur
++++

Defocus Blur
------------

Blur because the object is out of focus.


Motion Blur
-----------

- camera
    - shake during exposure
        - movement parallel to image plane
    - moving camera
        - Translation and Rotation
- moving objects

.. raw:: LaTex

    \newpage



Deblurring
++++++++++

Goal of deblurring: find blur kernel to get latent image

latent image + blur kernel = blurred image

Deconvolution
-------------

- non-blind
- semi-blind estimation with depth map or sensor data
- blind: blur kernel is unknown
    - point spread function (PSF) estimation


Different Approaches of deblurring
----------------------------------

- spatially kernel
    - invariant (uniform kernel)
    - variant (non-uniform kernel)

- input 
    - single image
        - sensor (intertial measurement)
        - motion density function
    - stereo image pairs
        - Hybrid Camera


Problems
--------

- noise in latent image
- artifacts in deblurred image

.. raw:: LaTex

    \newpage



Depth-Aware Motion Deblurring
+++++++++++++++++++++++++++++

Two view Stereopsis
-------------------

*Formulas*


Point Spread Function (PSF)
---------------------------

*Definition, Tree Approach*

This function is not robust on small regions, therefore larger regions are necessary.


Shock Filter
------------

cartoonize image: large regions with same color and clear object boundaries