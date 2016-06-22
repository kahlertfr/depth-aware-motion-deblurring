Deblurring a scene with depth variations blurred by translational camera
motion is no easy task. A deblurring algorithm assuming a uniform kernel
yields a poor result for this setup blurred in effect by non-uniform blur.
Figure :ref:`results` (a) shows the result of the two-phase kernel estimation
algorithm from Xu :cite:`Xu2010` assuming a uniform kernel. Whereas the
reference implementation based on the depth-aware motion deblurring algorithm
from Xu and Jia :cite:`Xu2012` yields a better result as presented in figure
:ref:`results` (b).

.. raw:: LaTex

    \begin{figure}[!htb]
        \centering
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=160pt]{../images/result-uniform-kernel.jpg}
            \caption{uniform result}
        \end{subfigure}%
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=160pt]{../images/deblur-left-irls.png}
            \caption{depth-aware result}
        \end{subfigure}
        \caption{Deblurring results of a uniform deblurring algorithm (two-phase kernel estimation algorithm) and the reference implementation as a depth-aware algorithm}
        \label{results}
    \end{figure}

Many difficulties had to be overcome in the implementation due to separate
handling of each depth layer. These layers produce arbitrary shaped regions
making it impossible to simply apply existing algorithms or methods for
further computation. Adjustments to the applied methods are always necessary
to handle the region boundary correct. Thus gradients of a region are computed
of the whole image and cropped to the region to avoid high gradients at region
boundaries. Besides that black areas of the top-level regions are filled to be
able to apply an existing PSF estimation algorithm which only works on
rectangular images. This technique avoids high gradients at the region
boundaries disturbing the PSF estimation too. Furthermore a region-wise
spatial deconvolution were implemented using mask support.

In the end the results of the paper can not be achieved by the current version
of the reference implementation. Many difficulties were solved but some
challenges remains like getting sparse PSF estimates from the mid-/leaf-level
PSF estimation. One reason are too many open questions for the details of the
single algorithm steps. This emphasizes the need for gaining access to the
source code making it more easier to fully understand the proposed algorithm
and reproduce the stated research results.

The results of the reference implementation may be improved by removing the
errors in the disparity maps leading to wrongly merged depth layers as
discussed in the previous chapter. Although the results of the paper are not
achieved by the reference implementation it contains several useful modules
like methods converted from matlab to C++ (conv2, corr and deconvolution
methods from Levin :cite:`Levin2007`).
