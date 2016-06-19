The reference implementation was used to deblur the same stereo image pair as presented in the paper. The reference implementation can not achieve the same results as shown in the paper. Hence this chapter discusses the problems leading to the different results.

Result Comparison
+++++++++++++++++

As shown in figure :ref:`result-compare` the results of the first iteration differ. The reference implementation generates more ringing artifacts resulting in brighter and darker regions. The result of the paper for the left view looks more smooth. The paper unfortunately does not present the result of the right view.

.. raw:: LaTex

    \begin{figure}[!htb]
        \centering
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/deblur-left-irls.png}
            \caption{reference implementation (left view)}
        \end{subfigure}%
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/deblur-right-irls.png}
            \caption{reference implementation (right view)}
        \end{subfigure}
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/mouse-result-1it-left-gray.jpg}
            \caption{result paper (left view)}
        \end{subfigure}
        \caption{Comparison results after 1. iteration}
        \label{result-compare}
    \end{figure}


The artifacts in the result of the reference implementation make it impossible to do a second iteration step since the "refined" disparity map computed from the deblurred views is worser than the initial disparity map of the blurred images. The disparity map of the deblurred views is shown in figure :ref:`dmap-2` using the same parameters for disparity estimation with graph-cut :cite:`Kolmogorov2001` as used in the first iteration. Even especially tuned parameters for this disparity estimation yielding a more smoothed result does not refine the initial disparity. 

.. raw:: LaTex

    \begin{figure}[!ht]
        \centering
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=115pt]{../images/dmap-final-left.png}
            \caption{initial}
        \end{subfigure}%
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=115pt]{../images/dmap-algo-left-2.png}
            \caption{2. run}
        \end{subfigure}%
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=115pt]{../images/dmap-algo-left-2-tuned.png}
            \caption{2. run (tuned)}
        \end{subfigure}
        \caption{disparity maps}
        \label{dmap-2}
    \end{figure}


Problem Discussion
++++++++++++++++++



**depth-layers**

- very small layers -> see figure :ref:`small-layers`
- :red:`some explanation`

.. raw:: LaTex

    \begin{figure}[!ht]
        \centering
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=100pt]{../images/mid-0-region-left.png}
            \caption{depth-layer 0}
        \end{subfigure}%
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=100pt]{../images/mid-3-region-left.png}
            \caption{depth-layer 3}
        \end{subfigure}%
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=100pt]{../images/mid-11-region-left.png}
            \caption{depth-layer 11}
        \end{subfigure}
        \caption{depth-layers with very small regions}
        \label{small-layers}
    \end{figure}


**psf estimation**

.. raw:: LaTex

    \begin{figure}[!ht]
        \centering
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=35pt]{../images/mid-5-kernel-selection-1.png}
            \caption{psf estimate}
        \end{subfigure}%
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=100pt]{../images/mid-5-region-left.png}
            \caption{region}
        \end{subfigure}%
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=100pt]{../images/mid-5-deconv-1-e0.191212.png}
            \caption{deconvolved region}
        \end{subfigure}

        \caption{example for blurry PSF estimate}
        \label{psf-estimate}
    \end{figure}

- psf estimates are very blurry -> see figure :ref:`psf-estimate` -> :red:`reason?`
- maybe they use a psf refinement step of their two-phase kernel estimation paper


**psf selection**

- the estimated kernels result in images with high contrast which are prefered by the psf selection scheme due to salient edges
- human eye would choose result of other kernel -> figure :ref:`wrong-select`

.. raw:: LaTex

    \begin{figure}[!ht]
        \centering
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=100pt]{../images/mid-10-deconv-0.png}
            \caption{chosen from algo}
        \end{subfigure}%
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=100pt]{../images/mid-10-deconv-1.png}
            \caption{prefered by human}
        \end{subfigure}

        \caption{top-level-regions (left view) and their PSFs (using two-phase kernel estimation executable)}
        \label{wrong-select}
    \end{figure}

**deblurring**

- final deconvolution: handling of different regions -> can see regions borders in my result


**influence deconvolution method**

.. raw:: LaTex

    \begin{figure}[!htb]
        \centering
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/deblur-left-fft.png}
            \caption{deconvolution using FFT}
        \end{subfigure}%
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/deblur-left-irls.png}
            \caption{deconvolution using IRLS}
        \end{subfigure}
        \caption{Influence of chosen deconvolution method (used within the algorithm process)}
        \label{result-deconv}
    \end{figure}

- child psf estimation used image deconvolved with parent psf
- psf selection deconvolves images
- results depends on chosen method -> figure :ref:`result-deconv`
- the paper doesn't mention how they do the deconvolution