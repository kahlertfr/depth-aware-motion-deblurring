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
- algorithm overview: disparity estimation, region tree construction, top-level PSF estimation, mid-level PSF refinement, deconv per depth layer and second run with refined disparity map

.. figure:: ../images/wip.png
   :width: 200 pt
   :alt: algorithm overview

   algorithm overview


Reference Implementation
++++++++++++++++++++++++

The reference implementation for the depth-aware motion deblurring algorithm provides a command line interface and a C++ library. A OpenCV 3.0 installation is required for this project. For further information please read the *README* of this project. The source code can be found online: :red:`add github repo`



Disparity Estimation
++++++++++++++++++++

- spatially variant kernel needed -> depth estimation with stereo matching

The main idea of the algorithm is the independent deblurring of each depth layer to get an accurate result for scenes with high depth differences. So the first step is the disparity estimation from both views.

Disparity Map
-------------

- Find disparity maps of a blurred stereo image pair: left to right and right to left
- user has to estimate the max disparity
- down-sampling for blur reducing
- minimizing energy function for each view (:red:`add variable explanation`)

.. math:: :numbered:
    
    E(d) = \| B_m(x - d(x)) - B_r(x)\|^2 + \gamma_d min(\nabla d^2, \tau)

- stereo algorithm: graph cut :cite:`Kolmogorov2001` -> their code is used
- used parameter values not mentioned, tuned by myself (max iterations set to 3)
- (result differs on same image because of random initialisation)
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
- finally upsampled

.. figure:: ../images/dmap-final-left.png
   :width: 200 pt
   :alt: disparity map quantized

   quantized disparity map with 12 regions (left view)



Region-Tree Construction
++++++++++++++++++++++++

- top-down estimation (from huge to small regions)
- in huge regions robust PSF estimation is possible
- in small regions PSF estimation is not robust: use parent PSF to guide PSF estimation

The regions of the different depth layer can be very small and therefore robust PSF estimation is not possible. The solution from Xu and Jia is a hierarchical estimation scheme where similar depth layers are merged to form larger regions. The structure for this is called region-tree and in the implementation it is the *RegionTree* class.

.. figure:: ../images/wip.png
   :width: 200 pt
   :alt: region tree

   12 quantized depth-layers result in 3 top-level regions

The region-tree is a binary tree with all depth layers as leaf nodes. Each mid or top level node is calculated the following way: depth layer S(i) and S(j) are merged if i and j are neighboring numbers and i = ⌊j/2⌋ * 2 which ensures that the neighbor of the current node is merged only once. If a node do not have any neighbor for merging the node becomes a top level node. This is done until the user specified number of top level nodes are reached.

The *RegionTree* class stores binary masks of all depth layer regions in the leaf nodes. The region of every other node can be computed by simply adding the masks of the regions that are contained in the current node.



PSF Estimation for Top-Level Regions
++++++++++++++++++++++++++++++++++++

- uses the two-phase kernel estimation algorithm of Xu :cite:`Xu2010`
- isn't implemented, as work-around: use provided exe to generate top-level PSFs (or any other kernel estimation algorithm)

.. raw:: LaTex

    \begin{figure}[!ht]
        \centering
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=100pt]{../images/top-0-left.jpg}
            \caption{background}
        \end{subfigure}%
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=100pt]{../images/top-1-left.jpg}
            \caption{middle}
        \end{subfigure}%
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=100pt]{../images/top-2-left.jpg}
            \caption{foreground}
        \end{subfigure}

        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=35pt]{../images/kernel0.png}
            \caption{background}
        \end{subfigure}%
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=35pt]{../images/kernel1.png}
            \caption{middle}
        \end{subfigure}%
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=35pt]{../images/kernel2.png}
            \caption{foreground}
        \end{subfigure}
        \caption{top-level-regions (left view) and their PSFs}
    \end{figure}

**problem**:

- regions are of arbitrary shape -> cannot crop image to get just the region
- region images have black pixel which do not belong to the region
- high gradients at borders of region would affect PSF estimation
- need for filling the pixel not belonging to the region in such a way that reduces high frequencies at the borders



PSF Propagation
+++++++++++++++


Blur Removal
++++++++++++

