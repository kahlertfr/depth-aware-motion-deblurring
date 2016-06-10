A lot of pictures are taken with a mobile phone or a hand-held camera. Blur caused by the shaking of the camera during the exposure is a widely spread problem. This blur in particular occurs when the camera moves during long exposure time such as in scenes with ambient illumination. It affects all pixels in the taken picture. In an image of a planar scene this blur is the same for all pixels. But scenes with different objects as shown in figure :ref:`tsu-image` results in different depths. The figure :ref:`tsu-gt` shows the different depth layers which are regions of nearly constant depth. In this case a near point is blurred more than a distant one. So each depth layer has its own blur kernel which is called spatially-variant blur.

.. raw:: LaTex

    \begin{figure}[!htb]
        \centering
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/tsukuba.jpg}
            \caption{scene with different objects}
            \label{tsu-image}
        \end{subfigure}%
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/tsukuba-gt.png}
            \caption{depth layers}
            \label{tsu-gt}
        \end{subfigure}
        \caption{3D-scene (tsukuba)}
    \end{figure}

The deblurring of these images is still an ongoing research issue. There are several approaches for removing blur from an image of a planar scene but less algorithms to remove spatially-variant blur. However the latter is interesting because most scenes have depth variations. It is hard to correctly deblur a depth scene without any depth information. Stereo image pairs can provide such information through stereo matching. Furthermore the necessary hardware to obtain stereo image pairs is more and more available – even in mobile phones.

Even with the depth information deblurring of scenes with depth variations isn't easy. Small-size regions lack necessary structural information for the blur kernel estimation. An approach to overcome this problem is presented in the paper from Xu and Jia :cite:`Xu2012`. They provide an iterative algorithm for motion deblurring with stereo images. It uses a spatially-varying point spread functions (PSF) to deblur the image on each depth level. They overcome the challenge for estimating the PSF in small-size regions with a hierarchical approach named region tree to guide this estimation.

This study thesis provides a reference implementation of this depth-aware motion deblurring algorithm from Xu and Jia. After a theoretical introduction to deblurring this algorithm is presented in detail. Indeed we will focus on the challenges that had to be solved to be able to deblur an image of a scene with depth variations.