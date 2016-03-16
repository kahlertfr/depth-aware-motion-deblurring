A short theoretical preliminary to deblurring.


Blur
++++

Blur is the result of averaging of intensities from different real world points in one image point. :cite:`CHO2009`

There are two major classes of blur: the defocus blur and the motion blur. The first one is obviously caused because an object is out of focus. The second one is caused by the relative motion between the camera and the scene during the exposure time.

The **motion blur** can occur due to different reasons: moving objects in the scene (such as vehicles) or camera movement. The camera could be moved freely in all directions of the room (including rotations) but we will focus on the camera movement parallel to the image plane because this is a common result of the shaking of the hands during the exposure. The result of such shift-invariant blur can be expressed in the following equation:

.. raw:: latex
    
    \begin{equation}
    B = I \otimes k + n
    \end{equation}

Where *B* is the observation, *I* is the latent (sharp) image convolved with a blur kernel *k* and *n* is some additional noise. :cite:`XU2012` The blur kernel is also known as point spread function (PSF) which describes the response of an imaging system to a point source or point object.



Deblurring
++++++++++

Deblurring is the task of finding the latent image.

The technique used for finding the latent image is called deconvolution. If the latent image and the blur kernel is unknown it is a blind deconvolution. In this case the PSF has to be estimated. Where as in the non-blind deconvolution the blur kernel is known or is assumed to be of an simple form.

The properties of the blur kernel vary: there are spatial invariant kernel also known as uniform kernel. They are used if the kernel in the image is everywhere the same. On the other hand there are spatially varying kernel also called non-uniform kernel which means that the kernel differs inside the image. This is the case of blurred images of depth scenes where each depth layer has its own kernel.

Deconvolving can be done in different ways.