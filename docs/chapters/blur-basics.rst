A short theoretical preliminary to deblurring.


Blur
++++

Blur is the result of averaged intensities from different real world points in one image point. :cite:`CHO2009`

There are two major classes of blur: the defocus blur and the motion blur. The first one is obviously caused because an object is out of focus. The second one is caused by the relative motion between the camera and the scene during the exposure time.

The **motion blur** can occur due to different reasons: moving objects in the scene (such as vehicles) or a moving camera. The camera could be moved freely in all directions of the room (including rotations) but we will focus on the camera movement parallel to the image plane because this is a common result of the shaking of the hands during the exposure. The result of such shift-invariant blur can be expressed in the following equation:

.. math:: :numbered:
    
    B = I \otimes k + n


Where *B* is the observation, *I* is the latent (sharp) image convolved with a blur kernel *k* and *n* is some additional noise. If the scene has different depths than there is a blur kernel for each depth layer. :cite:`XU2012` The blur kernel is also known as point spread function (PSF) which describes how an idealized point-shaped object is mapped through a system. So we can use it to describe the movement of a point on the image plane.

:red:`add images for PSF (point, kernel, blurred point)`



Deconvolution
+++++++++++++

Deblurring is the task of finding the latent image if a blurred image is given. The technique used for this is called deconvolution.

If the latent image and the blur kernel is unknown it is a blind deconvolution. In this case the PSF has to be estimated. Where as in the non-blind deconvolution the blur kernel is known or is assumed to be of an simple form.

The properties of the blur kernel vary: there are spatial invariant kernels also known as uniform kernels. They are used if the kernel in the image is everywhere the same. On the other hand there are spatially varying kernels also called non-uniform kernels which means that the kernel differs inside the image. This is the case in blurred images of depth scenes where each depth layer has its own kernel.

Deconvolution can be done in different ways: in the frequency domain or spatial domain.


Fourier Transformation
----------------------

The convolution of in the spatial domain can be expressed as an point-wise multiplication in the frequency domain in the following way:

.. math:: :numbered:
    
    I \otimes k  = iF(F(I) \times F(k))


Where an image *I* should be convolved with a kernel *k*. To transform the image and the kernel into the frequency domain the Fourier Transformation *F* is needed. To point-wise multiply the transformed kernel with the transformed image the kernel has to be of the same size as the image. To transform the result back into the spatial domain the inverse Fourier Transformation *iF* is used.

This can save a lot of time for the computation of the convolution. :cite:`SMITH2002` So it is worth it to have a look at the Fourier transformation.

:red:`add image for a function, the fourier sequence and the fourier transformation`

