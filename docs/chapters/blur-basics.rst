A short theoretical preliminary to deblurring.


Blur
++++

Blur is the result of averaged intensities from different real world points in one image point. :cite:`Cho2009`

There are two major classes of blur: the defocus blur and the motion blur. The first one is obviously caused because an object is out of focus. The second one is caused by the relative motion between the camera and the scene during the exposure time.

The **motion blur** can occur due to different reasons: moving objects in the scene (such as vehicles) or a moving camera. The camera could be moved freely in all directions of the room (including rotations) but we will focus on the camera movement parallel to the image plane because this is a common result of the shaking of the hands during the exposure. The result of such shift-invariant blur can be expressed in the following equation:

.. math:: :numbered:
    
    B = I \otimes k + n


Where *B* is the observation, *I* is the latent (sharp) image convolved with a blur kernel *k* and *n* is some additional noise. If the scene has different depths than there is a blur kernel for each depth layer. :cite:`Xu2012` The blur kernel is also known as point spread function (PSF) which describes how an idealized point-shaped object is mapped through a system. So we can use it to describe the movement of a point on the image plane.

.. raw:: LaTex

    \begin{figure}[!htb]
        \centering
        \includegraphics[width=120pt]{../images/psf-theory.png}
        \caption{Point-shaped objects convolved with a PSF and the result (Wikipedia)}
    \end{figure}


Deconvolution
+++++++++++++

Deblurring is the task of finding the latent image if a blurred image is given. The technique used for this is called deconvolution.

If the latent image and the blur kernel is unknown it is a blind deconvolution. In this case the PSF has to be estimated. Where as in the non-blind deconvolution the blur kernel is known or is assumed to be of an simple form.

The properties of the blur kernel vary: there are spatial invariant kernels also known as uniform kernels. They are used if the kernel in the image is everywhere the same. On the other hand there are spatially varying kernels also called non-uniform kernels which means that the kernel differs inside the image. This is the case in blurred images of depth scenes where each depth layer has its own kernel.

Deconvolution can be done in different ways: in the frequency domain or spatial domain.


Convolution Theorem
-------------------

The convolution theorem states that a convolution of in the spatial domain can be expressed as an point-wise multiplication in the frequency domain in the following way:

.. math:: :numbered:
    
    I \otimes k  = iF(F(I) \times F(k))


Where an image *I* should be convolved with a kernel *k*. The transformation of the image and the kernel into the frequency domain is done by using the Fourier Transformation *F*. The transformed kernel *F(k)* has to be of the same size as the image to be able to perform a point-wise multiplication. This could be done e.g. by copying the kernel into a black image with the size of the image *I* before the Fourier transformation. The position of the kernel in the black image doesn't matter because the Fourier transformation is shift-invariant. To transform the result back into the spatial domain the inverse Fourier Transformation *iF* is used.



Fourier Transformation
----------------------

The convolution theorem can save a lot of time for the computation of the convolution. So it is worth it to have a short look at the Fourier transformation.

.. raw:: LaTex

    \begin{figure}[!htb]
        \centering
        \includegraphics[width=220pt]{../images/fourier.jpg}
        \caption{Fourier Transformation (Wikipedia)}
    \end{figure}

A function *f(x)* (the red line in the figure) can be resolved as a linear combination of sines and cosines (the light blue functions in the figure) this is called a Fourier series. The following equation describes the Fourier series of a periodic function *f(x)* with period *N*:

.. math:: :numbered:
    
    f(x)  = \frac {a_0} {2} * \sum_k a_k cos( \frac {2 \pi kx} {N}) + \sum_k b_k sin( \frac {2 \pi kx} {N})
          = \sum_k c_k \rm{e}^{\rm{i} \frac {2 \pi kx} {N}}


The component frequencies of these sines and cosines result in peaks in the frequency domain (the dark blue function in the figure). The transformation of a function to these peaks in the frequency domain is called Fourier transformation.
In terms of image processing a discrete signal is given (the image) so the equations below describe the 2D discrete Fourier transformation (DFT). The technique for a fast computation of a discrete Fourier transformation is called Fast Fourier Transformation (FFT). :cite:`SMITH2002`

.. math:: :numbered:
    
    F(k,l)  = \sum_x \sum_y I(x,y) * \rm{e}^{-\rm{i} 2 \pi (\frac {kx} {C} + \frac{ly} {R})}

The next figure shows an example of the Fourier transformation of a horizontal cosine with 8 cycles and the second one is a vertical consine with 32 cycles. The result is the frequency coordinate system which center is in the center of the image.

.. raw:: LaTex

    \begin{figure}[!htb]
        \centering
        \includegraphics[width=150pt]{../images/cosines.jpg}
        \caption{Result of Fourier transformations of horizontal and vertical cosines}
    \end{figure}
