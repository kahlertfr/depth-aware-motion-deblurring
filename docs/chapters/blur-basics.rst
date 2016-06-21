In this chapter we give a short introduction to the blur formation model and the commonly used notations.

Blur
++++

Blur is the result of averaged intensities from different real world points in
one image point :cite:`Cho2009`. There are two major classes of blur: defocus
blur and motion blur.

Defocus blur
------------

Defocus blur is caused by the optics of the camera. Many factors such as lens
focal length and camera-to-subject distance can affect the focus range wherein
the objects are pictured sharply in the image. Objects that are out of focus
are blurred like the background and the near foreground in figure :ref:`d-b`.
The distance to the in-focus plane is related to the amount of blur. Objects
further away from the in-focus plane are more blurred in the image.

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

Motion blur is caused by relative motion between the camera and the scene
during long exposure times. This motion can occur due to different reasons:
object movement in the scene (such as vehicles or humans) or camera movement.
In images blurred by **object motion** as figure :ref:`m-b-o` each object is
affected by different blur. Hence segmentation of the objects is required for
deblurring.

Blur caused by **camera motion** depends on properties of the scene and the
camera movement. The simplest case is a flat scene and an in-plane camera
motion parallel to the scene which results in an image where every pixel is
affected by the same blur. That is also called uniform or spatially-invariant
blur. A scene with depth variations and an in-plane camera movement as figure
:ref:`m-b-c` results in an image where each depth layer is affected by
different blur :cite:`Xu2012`. Near objects are blurred more than distant
ones. In the case of in-plane camera movement the blur is scaled between the
depth layers. An arbitrary camera motion (rotation and translation) would
result in completely different blur for each depth layer. These are referred
to as non-uniform or spatially-variant blurs.

The camera motion parallel to the scene is more significant for handling blur
caused by shaking of hands during the exposure. In most cases the scene is
sufficiently far away to be able to disregard the effect of rotational motion
of the camera.



Blur as Convolution
+++++++++++++++++++

The blurred image *B* of a flat scene can be expressed as a convolution of a
sharp (latent) image *I* of this scene with a blur kernel *k*. Thus each pixel
of the scene is blurred with the same spatially-invariant blur kernel. Some
noise *n* have to be taken into account due camera related noise such as read-
out noise and shot noise. Although the noise is often neglected due to
simplification.

This blur can be expressed by the following equation:

.. math:: :numbered:
    
    B = I \otimes k + n

The amount of blur depends on the kernel size. An image convolved with a large
blur kernel is blurred more than one convolved with a small kernel.

The blur kernel is also known as Point Spread Function (PSF) which describes
how an idealized point-shaped object is mapped through a system
:cite:`SMITH2002`. So we can use it to describe the translational camera
movement parallel to the scene. The figure :ref:`psf-exp` shows a convolution
of a flat scene with a typical blur kernel caused by shaking of hands. These
kernels are usually very sparse.

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

If the scene is not flat but has different depths than there is a blur kernel
:math:`k^z` for each depth *z* thus this is a spatially-variant kernel
:cite:`Xu2012`.



Deblurring
++++++++++

Deblurring is the task of restoring the sharp image from a blurred one. It is
the inverse problem to the convolution of a sharp image with a blur kernel.
Thus the technique used for deblurring is called deconvolution. It can be
distinguished into non-blind deconvolution for a known blur kernel and blind
deconvolution for a unknown blur kernel.


Non-Blind Deconvolution
-----------------------

If the blur kernel is known or is assumed to be of a simple form then the
deconvolution is referred to as non-blind deconvolution.

Due to the reason that there is no mathematical inverse operation to
convolution some other techniques have to be used to perform a deconvolution.
One approach is to use the **convolution theorem** (see the corresponding
chapter) which transforms the problem into the frequency domain where the
deconvolution simply becomes a division. The Fourier transform *F* is used to
transform the blurred image *B* and the kernel *k* into the frequency domain.
The result is the sharp image in the frequency domain *F(I)*. To transform it
back to the spatial domain the inverse Fourier transform is needed. The
deconvolution in the frequency domain disregarding any noise is expressed in
the following equation:

.. math:: :numbered:
    
    F(I) = \frac{F(B)}{F(k)}

This approach is very fast because of efficient Fast Fourier Transform (FFT)
algorithms but is limited to a uniform kernel. This simple equation produces a
poor result because no noise is considered. Hence there are algorithms like
the Wiener deconvolution that works in the frequency domain but attempts to
minimize the affect of deconvolved noise by attenuating frequencies depending
on their signal-to-noise ratio :cite:`JAYA2009`.

There exists further approaches restoring the latent image blurred by an
uniform kernel in the spatial domain. Because the deconvolution is an ill-
posed problem and the solution may not be unique, the latent image can not be
computed directly. But iterative approaches like Richardson-Lucy deconvolution
try to find the most likely solution for the latent image :cite:`CAMPISI2007`.

.. raw:: LaTex


    \begin{figure}[!htb]
        \centering
        \begin{subfigure}{.25\textwidth}
            \centering
            \includegraphics[width=80pt]{../images/cm-original.jpg}
            \caption{original image}
        \end{subfigure}%
        \begin{subfigure}{.25\textwidth}
            \centering
            \includegraphics[width=80pt]{../images/cm-blurred.jpg}
            \caption{blurred image}
        \end{subfigure}%
        \begin{subfigure}{.25\textwidth}
            \centering
            \includegraphics[width=80pt]{../images/cm-w.jpg}
            \caption{Wiener}
        \end{subfigure}%
        \begin{subfigure}{.25\textwidth}
            \centering
            \includegraphics[width=80pt]{../images/cm-rl.jpg}
            \caption{Richardson-Lucy}
        \end{subfigure}
        \caption{Results of non-blind deconvolution with Wiener Deconvolution and Richardson-Lucy Deconvolution}
        \label{non-blind-deconv}
    \end{figure}

As shown in figure :ref:`non-blind-deconv` the restoration of a latent image
is not an easy task and the results of these simple approaches are not
satisfying. This motivates the research effort to find suitable models for a
better deconvolution which was presented in the related work chapter.

For spatially-variant kernels a segmentation into constant regions with the
same blur kernel is necessary. For motion blur caused by camera shake this
could be done using the depth map of a stereo image pair. Then the methods for
a uniform kernel can be applied to each region while taking care of region
boundaries to avoid visual artifacts.


Blind Deconvolution
-------------------

If the latent image and the blur kernel is unknown the deconvolution is
referred to as blind deconvolution. In this case the PSF has to be estimated.

The majority of blind deconvolution algorithm estimate the latent image and
the blur kernel simultaneously. For this a regularization framework is used
where the blind deblurring problem can be formulated as equation (3). *B* is
the blurred image, :math:`\tilde{I}` is the latent image, :math:`\tilde{k}` is
the blur kernel and :math:`\rho(I)` and :math:`\varrho(k)` are regularization
terms on the image and kernel :cite:`WANG2016`.

.. math:: :numbered:
    
    \{\tilde{I}, \tilde{k}\} = arg \min_{I,k} E(I,k) = arg \min_{I,k} ||I \otimes k - B ||_2^2 + \lambda \rho(I) + \gamma \varrho(k)

This equation minimizes the difference between the blurred image and the
latent image convolved with the blur kernel using the :math:`l^2`-norm while
considering assumption on the latent image and blur kernel expressed by
regularization terms. This again only holds for a uniform kernel.

The regularization terms are crucial to obtain better restoration results and
have to be chosen carefully. The regularization for the kernel is typically an
:math:`l^2`-norm penalty because small values distributed over the kernel are
preferred. Whereas the regularization term for the latent image is related to
the properties of natural images such as the existence of salient edges.

Finally the equation is solved by alternating between kernel estimation and
image estimation in an iterative way :cite:`CAMPISI2007`. Whereupon kernel
estimation results depend heavily on the image texture. In regions of no
texture any blur kernel is possible because blurring a homogeneous region do
not affect the region at all.

As before spatially-variant blur has to be estimated for regions of nearly
equal blur seperately.


Convolution Theorem
-------------------

The convolution theorem states that a convolution of an image *I* with a
kernel *k* in the spatial domain can be expressed as an point-wise
multiplication in the frequency domain :cite:`SMITH2002`. The transformation
of the image and the kernel into the frequency domain is done by using the
Fourier transform *F*. For the backwards transformation into the spatial
domain the inverse Fourier transform :math:`F^{-1}` is used.

This theorem only holds for a uniform kernel and is expressed by the following
equation where :math:`\times` is the point-wise multiplication:

.. math:: :numbered:
    
    I \otimes k  = F^{-1}(F(I) \times F(k))


The transformed kernel *F(k)* has to be of the same size as the image to be
able to perform a point-wise multiplication.



.. Fourier Transformation
.. ----------------------

.. The convolution theorem can save a lot of time for the computation of the
.. convolution. So it is worth it to have a short look at the Fourier
.. transformation.

.. .. raw:: LaTex

..     \begin{figure}[!htb]
..         \centering
..         \includegraphics[width=220pt]{../images/fourier.jpg}
..         \caption{Fourier Transformation (Wikipedia)}
..     \end{figure}

.. A function *f(x)* (the red line in the figure) can be resolved as a linear
.. combination of sines and cosines (the light blue functions in the figure)
.. this is called a Fourier series. The following equation describes the
.. Fourier series of a periodic function *f(x)* with period *N*:

.. .. math:: :numbered:
    
..     f(x)  = \frac {a_0} {2} * \sum_k a_k cos( \frac {2 \pi kx} {N}) + \sum_k b_k sin( \frac {2 \pi kx} {N})
..           = \sum_k c_k \rm{e}^{\rm{i} \frac {2 \pi kx} {N}}


.. The component frequencies of these sines and cosines result in peaks in the
.. frequency domain (the dark blue function in the figure). The transformation
.. of a function to these peaks in the frequency domain is called Fourier
.. transformation. In terms of image processing a discrete signal is given
.. (the image) so the equations below describe the 2D discrete Fourier
.. transformation (DFT). The technique for a fast computation of a discrete
.. Fourier transformation is called Fast Fourier Transformation (FFT)
.. :cite:`SMITH2002`.

.. .. math:: :numbered:
    
..     F(k,l)  = \sum_x \sum_y I(x,y) * \rm{e}^{-\rm{i} 2 \pi (\frac {kx} {C} + \frac{ly} {R})}

.. The next figure shows an example of the Fourier transformation of a
.. horizontal cosine with 8 cycles and the second one is a vertical consine
.. with 32 cycles. The result is the frequency coordinate system which center
.. is in the center of the image.

.. .. raw:: LaTex

..     \begin{figure}[!htb]
..         \centering
..         \includegraphics[width=150pt]{../images/cosines.jpg}
..         \caption{Result of Fourier transformations of horizontal and vertical cosines}
..     \end{figure}
