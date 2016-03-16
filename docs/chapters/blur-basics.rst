A short theoretical preliminary to deblurring and the used notation within this study thesis.


Blur
++++

Blur is the result of averaging of intensities from different real world points in one image point. 

There are two major classes of blur: the defocus blur and the motion blur. The first one is obviously caused because an object is out of focus. The second one is caused by the relative motion between the camera and the scene during the exposure time.

The **motion blur** can occur due to different reasons: moving objects in the scene (such as vehicles) or camera movement. The camera could be moved freely in all directions of the room (including rotations) but we will focus on the camera movement parallel to the image plane because this is a common result of the shaking of the hands during the exposure. The result of such blur can be expressed in the following equation:

.. raw:: latex
    
    \begin{equation}
    B = I \otimes k + n
    \end{equation}

Where *B* is the observation, *I* is the latent (sharp) image, *k* is the blur kernel and *n* is some additional noise. The blur kernel is also known as point spread function (PSF) which describes the response of an imaging system to a point source or point object.

