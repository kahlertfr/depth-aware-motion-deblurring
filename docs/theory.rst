.. section-numbering::
    :depth: 2

======
Theory
======

Depth-Aware Motion Blur
=======================

++++++++++++++++++++++
General Theory of Blur
++++++++++++++++++++++

Blur
++++

Blur is the result of averaging of intensities from different real world point in on image point.


Defocus Blur
------------
Blur because the object is out of focus.


Motion Blur
-----------
Blur because of relative motion between camera and a scene during exposure time.

- camera
    - shake during exposure
        - movement parallel to image plane
    - moving camera
        - Translation and Rotation
- moving objects



Deblurring
++++++++++

Goal of deblurring: find blur kernel to get latent image

latent image + blur kernel = blurred image

Deconvolution
-------------

- non-blind
    - blur kernel is known or assumed to be of a simple form (uniform camera motion)
    - try to reduce artifacts
- blind
    - blur kernel and latent image is unknown
    - point spread function (PSF) estimation


Different Approaches of deblurring
----------------------------------

- spatially kernel
    - invariant (uniform kernel)
    - variant (non-uniform kernel)

- input 
    - single image
        - additional devices, e.g. sensor (intertial measurement)
        - without additional hardware, e.g. motion density function
    - stereo image pairs
        - Stereo camera
        - Hybrid camera


Problems
--------

- noise in latent image
- artifacts in deblurred image
    - ringing artifacts at strong egdes

.. raw:: LaTex

    \newpage



+++++++++++++++++++++++++++++
Depth-Aware Motion Deblurring
+++++++++++++++++++++++++++++

Two view Stereopsis
+++++++++++++++++++

*Formulas*


Point Spread Function (PSF)
+++++++++++++++++++++++++++

*Definition, Tree Approach*

This function is not robust on small regions, therefore larger regions are necessary.


Shock Filter
++++++++++++

cartoonize image: large regions with same color and clear object boundaries