**setup**

- stereo image pair of a depth scene

**challenges**

- unkown motion of camera (parallel to scene)
- depth -> spatially variant PSF
- small number of pixels to estimate PSF from

The depth-aware motion deblurring algorithm from Xu and Jia :cite:`Xu2012` deals with this challenging setup. The basic idea of this paper and the challenges of its algorithm will be presented in this chapter.


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



Basic Idea
++++++++++

- depth information from stereo matching
- estimate PSF for each depth level
- constructing region tree to guide PSF estimation in small regions (similar PSFs for closeby depth levels)

.. figure:: ../images/wip.png
   :width: 200 pt
   :alt: algorithm overview

   algorithm overview


Reference Implementation
++++++++++++++++++++++++

The reference implementation for the depth-aware motion deblurring algorithm provides a command line interface and a C++ library. A OpenCV 3.0 installation is required for this project. For further information please read the *README* of this project. The source code can be found online: https://square-src.de/gitlab/franzi/study-thesis.git



Disparity Estimation
++++++++++++++++++++

The main idea of the algorithm is the independent deblurring of each depth layer to get an accurate result for scenes with high depth differences. So the first step is the disparity estimation from both views.

Disparity Map
-------------

- Find disparity maps of a blurred stereo image pair: left to right and right to left
- user has to estimate the max disparity
- down-sampling for blur reducing
- stereo algorithm: graph cut :cite:`Kolmogorov2001` -> their code is used
- alternative stereo matching algorithm also implemented: SGBM :cite:`Hi2007` 

- :red:`violation of stereo matching condition? handle boundary pixel separately -> how? not mentioned in paper`


Occlusions
----------

- Cross-Checking to find occluded regions
- using code from :cite:`Kolmogorov2001`

Occlusions are filled with smallest neighbor disparity. Assumption: just objects with small
disparity can be occluded.

.. raw:: LaTex

    \begin{figure}[!ht]
        \centering
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/dmap-algo-left.png}
            \caption{left-right}
        \end{subfigure}%
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/dmap-algo-right.png}
            \caption{right-left}
        \end{subfigure}
        \caption{disparity maps with filled occlusions}
    \end{figure}


Quantization
------------

- PSF estimation is less extensive if the disparity layers are reduced
- quantize disparity values to l regions, where l is set to approximate PSF width or height -> in practice 12 layers are enough (from paper)
- using k-means for clustering (both maps together to get same clusters for same depth)
- sort clusters for representing depth graduation

.. figure:: ../images/dmap-final-left.png
   :width: 200 pt
   :alt: disparity map quantized

   quantized disparity map with 12 regions (left view)



Region-Tree Construction
++++++++++++++++++++++++

The regions of the different depth layer can be very small and therefore robust PSF estimation is not possible. The solution from Xu and Jia is a hierarchical estimation scheme where similar depth layers are merged to form larger regions. The structure for this is called region-tree and in the implementation it is the *RegionTree* class.

.. figure:: ../images/wip.png
   :width: 200 pt
   :alt: region tree

   region tree

The region-tree is a binary tree with all depth layers as leaf nodes. Each mid or top level node is calculated the following way: depth layer S(i) and S(j) are merged if i and j are neighboring numbers and i = ⌊j/2⌋ * 2 which ensures that the neighbor of the current node is merged only once. If a node do not have any neighbor for merging the node becomes a top level node. This is done until the user specified number of top level nodes are reached.

The *RegionTree* class stores binary masks of all depth layers regions in the leaf nodes. The region of every other node can be computed by simply adding the masks of the regions that are contained in the current node.



PSF Estimation for Top-Level Regions
++++++++++++++++++++++++++++++++++++

This follows the algorithm of :cite:`Xu2010`.

:red:`tried to implement the two-phase kernel estimation` but unfortunately this couldn't be finished in time. So the provided exe is used to generate the top-level PSFs which are necessary to go on with the main algorithm.

- :red:`used edge tapering for region images to reduce high frequencies at the borders of the regions - so initial PSF estimation for the top level regions can be done with any kernel estimation algorithm`



PSF Propagation
+++++++++++++++


Blur Removal
++++++++++++

