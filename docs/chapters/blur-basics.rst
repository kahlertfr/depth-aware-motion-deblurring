We give a short introduction to the blur formation model and the commonly used notations in this chapter.

Blur
++++

Blur is the result of averaged intensities from different real world points in one image point :cite:`Cho2009`. There are two major classes of blur: defocus blur and motion blur.

Defocus blur
------------

Defocus blur is caused by the optics of the camera. Many factors such as lens focal length and camera-to-subject distance can affect the focus range wherein the objects are sharp. Objects that are out of focus are blurred as the background and the near foreground in figure :ref:`d-b`. The distance to the in-focus plane is related to the amount of blur. Objects further away to the in-focus plane are more blurred in the image.

.. raw:: LaTex

    \begin{figure}[!htb]
        \centering
        \begin{subfigure}{.33\textwidth}
            \centering
            \includegraphics[height=85pt]{../images/defocus-office.jpg}
            \caption{defocus blur}
            \label{d-b}
        \end{subfigure}%
        \begin{subfigure}{.33\textwidth}
            \centering
            \includegraphics[height=85pt]{../images/motion-blur-object.jpg}
            \caption{object motion}
            \label{m-b-o}
        \end{subfigure}%
        \begin{subfigure}{.33\textwidth}
            \centering
            \includegraphics[height=85pt]{../images/mouse_right.jpg}
            \caption{camera motion}
            \label{m-b-c}
        \end{subfigure}
        \caption{Examples of blurred images}
    \end{figure}


Motion blur
-----------

Motion blur is caused by the relative motion between the camera and the scene during long exposure times. This motion can occur due to different reasons: object movement in the scene (such as vehicles or humans) or camera movement. In images blurred by **object motion** as figure :ref:`m-b-o` each object is affected by different blur. Hence segmentation of the objects is required for deblurring.

Blur caused by **camera motion** depends on properties of the scene and the camera movement. The simplest case is a flat scene and an in-plane camera motion parallel to the scene which results in an image where every pixel is affected by the same blur. That is also called uniform or spatially-invariant blur. A scene with depth variations as figure :ref:`m-b-c` and an in-plane camera movement results in an image where each depth layer is affected by different blur :cite:`Xu2012`. This blur is scaled between the depth layers. An arbitrary camera motion (rotation and translation) would result in completely different blur for each depth layer. These are referred to as non-uniform or spatially-variant blurs. 

The camera motion parallel to the scene is more significant to handle blur caused by shaking of the hands during the exposure. This is because in most cases the scene is sufficiently far away to be able to disregard the effect of rotational motion of the camera.



Blur as Convolution
+++++++++++++++++++

The blurred image *B* of a flat scene can be expressed as a convolution of a sharp (latent) image *I* of this scene with a blur kernel *k*. Such that each pixel of the scene is blurred with the same spatially-invariant blur kernel. Some noise *n* have to be taken into account due camera related noise such as read-out noise and shot noise. Although the noise is often neglected due to simplification.

This blur can be expressed by the following equation:

.. math:: :numbered:
    
    B = I \otimes k + n


If the scene is not flat but has different depths than there is a blur kernel :math:`k^z` for each depth *z* thus this is a spatially-variant kernel.

The blur kernel is also known as point spread function (PSF) which describes how an idealized point-shaped object is mapped through a system. So we can use it to describe the movement of a point on the image plane. The figure :ref:`psf-exp` shows a convolution of a flat scene with a typical hand-shake blur kernel.

.. raw:: LaTex


    \begin{figure}[!htb]
        \centering
        \begin{subfigure}{.3\textwidth}
            \centering
            \includegraphics[width=110pt]{../images/image.png}
            \caption{scene}
        \end{subfigure}%
        \begin{subfigure}{.3\textwidth}
            \centering
            \includegraphics[width=30pt]{../images/kernel.png}
            \caption{PSF}
        \end{subfigure}%
        \begin{subfigure}{.3\textwidth}
            \centering
            \includegraphics[width=110pt]{../images/conv.png}
            \caption{result}
        \end{subfigure}
        \caption{Flat scene with arbitrary objects convolved with a typical hand-shake PSF}
        \label{psf-exp}
    \end{figure}



Deblurring
++++++++++
Deblurring is the task of finding the sharp image of a blurred one. It is the inverse problem to the convolution of a sharp image with a blur kernel. Thus the technique used for this is called deconvolution. It can be distingu


Non-Blind Deconvolution
-----------------------

The blur kernel is known or is assumed to be of a simple form then the deconvolution is referred to as non-blind deconvolution.

Due to the reason that mathematically there is no inverse operation to convolution some other techniques have to be used to perform a deconvolution. One approach is using the **convolution theorem** (see the corresponding chapter) which transforms the problem into the frequency domain where the deconvolution simply becomes a division. The Fourier Transformation *F* is used to transform the blurred image *B* and the kernel *k* into the frequency domain. The result is the sharp image in the frequency domain *F(I)*. To transform it back to the spatial domain the inverse Fourier Transformation is needed.

.. math:: :numbered:
    
    F(I) = \frac{F(B)}{F(k)}

This approach is very fast because of efficient Fast Fourier Transformation (FFT) algorithms but is limited to a uniform kernel.

:red:`TODO: spatial approach`

For spatially-variant kernels a segmentation into constant regions where each kernel has to be applied is necessary. This could be done using depth maps of stereo image pairs for motion blur. Then the methods for a uniform kernel can be applied to each region.


Blind Deconvolution
-------------------

If the latent image and the blur kernel is unknown it is a blind deconvolution. In this case the PSF has to be estimated.

:red:`TODO: write something`

- estimate kernel and image iteratively
- importance of texture


Convolution Theorem
-------------------

The convolution theorem states that a convolution of an image *I* with a kernel *k* in the spatial domain can be expressed as an point-wise multiplication in the frequency domain. The transformation of the image and the kernel into the frequency domain is done by using the Fourier Transformation *F*. For the transformation back into the spatial domain the inverse Fourier Transformation *iF* is used.

This theorem only holds for a uniform kernel and is expressed by the following equation where *x* is the point-wise multiplication:

.. math:: :numbered:
    
    I \otimes k  = iF(F(I) \times F(k))


The transformed kernel *F(k)* has to be of the same size as the image to be able to perform a point-wise multiplication.



.. Fourier Transformation
.. ----------------------

.. The convolution theorem can save a lot of time for the computation of the convolution. So it is worth it to have a short look at the Fourier transformation.

.. .. raw:: LaTex

..     \begin{figure}[!htb]
..         \centering
..         \includegraphics[width=220pt]{../images/fourier.jpg}
..         \caption{Fourier Transformation (Wikipedia)}
..     \end{figure}

.. A function *f(x)* (the red line in the figure) can be resolved as a linear combination of sines and cosines (the light blue functions in the figure) this is called a Fourier series. The following equation describes the Fourier series of a periodic function *f(x)* with period *N*:

.. .. math:: :numbered:
    
..     f(x)  = \frac {a_0} {2} * \sum_k a_k cos( \frac {2 \pi kx} {N}) + \sum_k b_k sin( \frac {2 \pi kx} {N})
..           = \sum_k c_k \rm{e}^{\rm{i} \frac {2 \pi kx} {N}}


.. The component frequencies of these sines and cosines result in peaks in the frequency domain (the dark blue function in the figure). The transformation of a function to these peaks in the frequency domain is called Fourier transformation.
.. In terms of image processing a discrete signal is given (the image) so the equations below describe the 2D discrete Fourier transformation (DFT). The technique for a fast computation of a discrete Fourier transformation is called Fast Fourier Transformation (FFT) :cite:`SMITH2002`.

.. .. math:: :numbered:
    
..     F(k,l)  = \sum_x \sum_y I(x,y) * \rm{e}^{-\rm{i} 2 \pi (\frac {kx} {C} + \frac{ly} {R})}

.. The next figure shows an example of the Fourier transformation of a horizontal cosine with 8 cycles and the second one is a vertical consine with 32 cycles. The result is the frequency coordinate system which center is in the center of the image.

.. .. raw:: LaTex

..     \begin{figure}[!htb]
..         \centering
..         \includegraphics[width=150pt]{../images/cosines.jpg}
..         \caption{Result of Fourier transformations of horizontal and vertical cosines}
..     \end{figure}
