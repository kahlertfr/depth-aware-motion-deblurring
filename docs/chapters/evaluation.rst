- run reference implementation on same images as the paper -> cannot achieve the same results

.. raw:: LaTex

    \begin{figure}[!htb]
        \centering
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/wip.png}
            \caption{result of paper}
        \end{subfigure}%
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/wip.png}
            \caption{result of reference implementation}
        \end{subfigure}
        \caption{Comparison paper results with reference implementation}
    \end{figure}


Problem Discussion
++++++++++++++++++

**psf estimation**

- psf estimates are very blurry -> :red:`reason?`
- maybe they use a psf refinement step of their two-phase kernel estimation paper

:red:`TODO: more!`