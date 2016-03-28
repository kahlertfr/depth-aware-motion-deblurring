The depth-aware motion deblurring algorithm was developed by Xu and Jia :cite:`XU2012`. This algorithm operates on a stereo image pair and processes the images with the steps described below. For improving the result the process is done a second time with the deblurred images from the first run as input for the second one.

.. raw:: LaTex

    \begin{figure}[!htb]
        \centering
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/mouse_left.jpg}
            \caption{left image}
        \end{subfigure}%
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/mouse_right.jpg}
            \caption{right image}
        \end{subfigure}
        \caption{Blurred input images}
    \end{figure}



Reference Implementation
++++++++++++++++++++++++

The reference implementation for the depth-aware motion deblurring algorithm provides a command line interface and a C++ library. A OpenCV 3.0 installation is required for this project. For further information please read the *README* of this project. The source code can be found here: https://square-src.de/gitlab/franzi/study-thesis.git

The project contains two independent algorithms: the two-phase kernel estimation algorithm from Xu and Jia :cite:`XU2010` and the depth-aware motion deblurring algorithm. The first one is used inside for the depth-aware deblurring. Because both algorithms are independent there is also a standalone command line interface and a C++ library for the two-phase kernel estimation algorithm.
:red:`Unfortunately the two-phase kernel estimation algorithm could not be finished within the context of this study thesis.`


Disparity Estimation
++++++++++++++++++++

The main idea of the algorithm is the independent deblurring of each depth layer to get an accurate result for scenes with high depth differences. So the first step is the disparity estimation from both views.

Disparity Map
-------------

- :red:`Find disparity maps of a blurred stereo image pair: left to right and right to left`
- :red:`down-sampling for blur reducing`
- :red:`different stereo algorithm as in paper. This shouldn't effect overall result.` Using SGBM :cite:`Hi2007`
- :red:`comments on SGBM parameters: choose of regularization term for smoothing, min disparity?`
- :red:`right to left: flip images such that SGBM works`
- :red:`violation of stereo matching condition. handle boundary pixel separately`


Occlusions
----------

:red:`Cross-Checking to find occlusion regions.` In this implementation there is no cross checking
because SGBM handles occluded regions already.

Occlusions are filled with smallest neighbor disparity. Assumption: just objects with small
disparity can be occluded.

.. raw:: LaTex

    \begin{figure}[!ht]
        \centering
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/dmap_small.jpg}
            \caption{with occlusions}
        \end{subfigure}%
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/dmap_small_filled.jpg}
            \caption{with filled occlusions}
        \end{subfigure}
        \caption{disparity map}
    \end{figure}


Quantization
------------

:red:`PSF estimation is less extensive if the disparity layers are reduced.` quantize disparity 
values to l regions, where l is set to approximate PSF width or height. :red:`how to approximate
the PSF width/height?`

- :red:`using k-means for clustering`
- :red:`sort clusters for representing depth graduation`

.. figure:: ../images/dmap_final.jpg
   :width: 200 pt
   :alt: disparity map quantized

   quantized disparity map with 25 regions



Region-Tree Construction
++++++++++++++++++++++++

The regions of the different depth layer can be very small and therefore robust PSF estimation is not possible. The solution from Xu and Jia is a hierarchical estimation scheme where similar depth layers are merged to form larger regions. The structure for this is called region-tree and in the implementation it is the *RegionTree* class.

The region-tree is a binary tree with all depth layers as leaf nodes. Each mid or top level node is calculated the following way: depth layer S(i) and S(j) are merged if i and j are neighboring numbers and i = ⌊j/2⌋ * 2 which ensures that the neighbor of the current node is merged only once. If a node do not have any neighbor for merging the node becomes a top level node. This is done until the user specified number of top level nodes are reached.

The *RegionTree* class stores binary masks of all depth layers regions in the leaf nodes. The region of every other node can be computed by simply adding the masks of the regions that are contained in the current node.



PSF Estimation for Top-Level Regions
++++++++++++++++++++++++++++++++++++

This follows the algorithm of :cite:`XU2010`.

:red:`tried to implement the two-phase kernel estimation` but unfortunately this couldn't be finished in time. So the provided exe is used to generate the top-level PSFs which are necessary to go on with the main algorithm.

- :red:`used edge tapering for region images to reduce high frequencies at the borders of the regions - so initial PSF estimation for the top level regions can be done with any kernel estimation algorithm`



PSF Propagation
+++++++++++++++


Blur Removal
++++++++++++

