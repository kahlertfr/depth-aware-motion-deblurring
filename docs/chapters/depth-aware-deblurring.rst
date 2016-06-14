**setup**

- stereo image pair of a depth scene -> figure :ref:`input`

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
            \caption{left image (reference view)}
        \end{subfigure}%
        \begin{subfigure}{.5\textwidth}
            \centering
            \includegraphics[width=170pt]{../images/mouse_right.jpg}
            \caption{right image (matching view)}
        \end{subfigure}
        \caption{Blurred input images}
        \label{input}
    \end{figure}



Basic Idea
++++++++++

- depth information from stereo matching
- estimate PSF for each depth level
- constructing region tree to guide PSF estimation in small regions (similar PSFs for close by depth levels)
- algorithm overview: disparity estimation, region tree construction, top-level PSF estimation, mid-level PSF refinement, deconv per depth layer and second run with refined disparity map -> figure :ref:`algo`

.. figure:: ../images/algo-overview.jpg
   :width: 100%

   :label:`algo` algorithm overview



Reference Implementation
++++++++++++++++++++++++

The reference implementation for the depth-aware motion deblurring algorithm provides a command line interface and a C++ library. A OpenCV 3.0 installation is required for this project. For further information please read the *README* of this project. The source code can be found online: :red:`add github repo`



Disparity Estimation
++++++++++++++++++++

- spatially variant kernel dependent on depth -> depth estimation with stereo matching

The main idea of the algorithm is the independent deblurring of each depth layer to get an accurate result for scenes with high depth differences. So the first step is the disparity estimation from both views.

Disparity Map
-------------

- Find disparity maps of a blurred stereo image pair: left to right and right to left
- down-sampling for blur reducing
- minimizing energy function for each view (:red:`add variable explanation`)

.. math:: :numbered:
    
    E(d) = \| B_m(x - d(x)) - B_r(x)\|^2 + \gamma_d min(\nabla d^2, \tau)

- stereo algorithm: graph cut :cite:`Kolmogorov2001` -> their code is used
- used parameter values not mentioned, tuned by myself (max iterations set to 3)
- (result differs on same image because of random initialisation)
- alternative stereo matching algorithm also implemented: SGBM :cite:`Hi2007`


**problems**

- general problem of occlusion (no correct object borders) -> affects all following steps (mainly deblurring)
- handle region boundary pixels separately (e.g. in deblurring with adjusted weight)
- finally second run to refine dmaps to get correct object boundaries


Occlusions
----------

- Cross-Checking to find occluded regions
- using code from :cite:`Kolmogorov2001` -> result figure :ref:`dmap-algo`

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
        \label{dmap-algo}
    \end{figure}


Quantization
------------

- PSF estimation is less extensive if the disparity layers are reduced
- quantize disparity values to l regions, where l is set to approximate PSF width or height -> in practice 12 layers are enough (from paper)
- using k-means for clustering (both maps together to get same clusters for same depth)
- sort clusters for representing depth graduation -> see figure :ref:`dmap-quant`
- finally up-sampled

.. figure:: ../images/dmap-final-left.png
   :width: 200 pt

   :label:`dmap-quant` quantized disparity map with 12 regions (left view)



Region-Tree Construction
++++++++++++++++++++++++

The regions of the different depth layer can be very small and therefore robust PSF estimation is not possible. The solution from Xu and Jia is a hierarchical estimation scheme where similar depth layers are merged to form larger regions. The structure for this is called region-tree and in the implementation it is the *RegionTree* class.

- top-down estimation (from huge to small regions)
- in huge regions robust PSF estimation is possible
- in small regions PSF estimation is not robust: use parent PSF to guide PSF estimation

.. figure:: ../images/regiontree-detail.jpg
   :width: 300 pt

   :label:`regiontree` one part of the regiontree where the depth layers 4-7 are merged together to one top-level node

The region-tree is a binary tree with all depth layers as leaf nodes. Each mid or top level node is calculated the following way: depth layer S(i) and S(j) are merged if i and j are neighboring numbers and i = ⌊j/2⌋ * 2 which ensures that the neighbor of the current node is merged only once. If a node do not have any neighbor for merging the node becomes a top level node. This is done until the user specified number of top level nodes are reached. The result is shown in figure :ref:`regiontree`.

The *RegionTree* class stores binary masks of all depth layer regions in the leaf nodes. The region of every other node can be computed by simply adding the masks of the regions that are contained in the current node.

**problem**

- some regions are very small and haven't any texture in them



PSF Estimation for Top-Level Regions
++++++++++++++++++++++++++++++++++++

- uses the two-phase kernel estimation algorithm of Xu :cite:`Xu2010`
- isn't implemented, as work-around: use provided exe to generate top-level PSFs (or any other kernel estimation algorithm)
- results of the two-phase kernel estimation algo for top-level regions see figure :ref:`top-level`

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
        \caption{top-level-regions (left view) and their PSFs (using two-phase kernel estimation executable)}
        \label{top-level}
    \end{figure}

**problem (implementation)**:

- regions are of arbitrary shape -> cannot crop image to get just the region
- region images have black pixel which do not belong to the region
- high gradients at borders of region would affect PSF estimation
- two possibilities: mask support (only consider pixel inside region) or fill the pixel not belonging to the region in such a way that reduces high frequencies at the borders (edge tapering)



Iterative PSF Computation
+++++++++++++++++++++++++

- for mid- and leaf level nodes
- regions become smaller and smaller on the way from top to bottom in the region tree -> PSF estimation isn't robust
- parent PSF estimate is available to guide child PSF estimation
- because of erroneous estimates in very small regions a PSF selection scheme is provided
- lack of texture is a problem too - handled by candidate selection
- the two steps of iterative PSF computation for each node is described below

.. figure:: ../images/mid-level-estimation.jpg
   :width: 170 pt

   :label:`mid-est` A PSF selection process for the current mid/leaf-level node (yellow one) containing given parent PSF, intial PSF estimation for current node and sibbling node, candidate selection and finally PSF selection


Joint PSF Estimation
--------------------

- guide estimation with salient edge map :math:`\nabla S`
    - parent PSF is used to compute the edge map
    - same as P map from Fast Motion Deblurring :cite:`Cho2009` (deblur with parent, bilateral filter, shock filter, gradients)
- Tikhonov regularization (here L2 regularization for k -> sparsity of kernel)
- :red:`add variable explanation for coming formulas`
- objective function is defined jointly on reference and matching view (more robust against noise)

.. math:: :numbered:
    
    E(k) = \sum_{i \in \{r,m\}} \| \nabla S_i \otimes k - \nabla B_i \|^2 + \gamma_k \|k\|^2

- closed-form solution using Fourier Transformations

.. math:: :numbered:
    
    k = F^{-1} \frac
        {\sum_i \overline{F_{\partial_x S_i}} F_{\partial_x B_i}  +  \sum_i \overline{F_{\partial_y S_i}} F_{\partial_ y B_i}} 
        {\sum_i (\overline{F_{\partial_x S_i}} F_{\partial_x S_i} + \overline{F_{\partial_y S_i}} F_{\partial_y S_i} )  +  \gamma_k F_{1}^2}

**problem**:

- gradients of regions: border of region results in huge gradient therefore compute gradients always on the whole image and then cut the region
- same problem appears if the gradient is calculated in Fourier domain -> vary formula of paper to compute gradients of region in spatial to domain to be able to cut of the region


Candidate PSF Selection
-----------------------

- major novelty of this paper
- PSF estimate can be erroneous -> detect incorrect PSFs (mostly very noisy and dense values)
- PSF entropy

.. math:: :numbered:

    H(k) = - \sum_{x \in k} x \log x

- mark PSF as unreliable if entropy is notably larger than it peers in the same level (through all three sub-trees)

- candidates are: parent and own kernel and sibling kernel if reliable

**problem**:

- PSF candidates available but how to determine what deconvolution has the best result
- new PSF selection scheme proposed: a correct deblurred image should contain salient edges
- salient edges are invariant to shock filtering that means they won't be affected -> compare deblurred image with its shock filtered version to check for salient edges
- (the requirement of salient edges in latent image is mostly satisfied)

**details of psf selection scheme**

- restore latent image :math:`I^k` for each kernel candidate

.. math:: :numbered:

    E(I^k) = \| I^k \otimes k - B \|^2 +  \gamma \|\nabla I^k \|^2


.. raw:: LaTex

    \begin{figure}[!ht]
        \centering
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=35pt]{../images/mid-2-kernel-init.png}
            \caption{ estimated PSF}
        \end{subfigure}%
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=35pt]{../images/kernel0.png}
            \caption{ PSF from parent}
        \end{subfigure}%
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=35pt]{../images/mid-3-kernel-init.png}
            \caption{ PSF from sibbling}
        \end{subfigure}

        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=100pt]{../images/mid-2-deconv-0.png}
            \caption{energy 0.19057}
        \end{subfigure}%
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=100pt]{../images/mid-2-deconv-1.png}
            \caption{energy 0.19255}
        \end{subfigure}%
        \begin{subfigure}{.35\textwidth}
            \centering
            \includegraphics[width=100pt]{../images/mid-2-deconv-2.png}
            \caption{energy 0.19733}
        \end{subfigure}
        \caption{PSF selection for one node with 3 candidates and the deconvolved images. The candidate with the smallest energy is chosen}
        \label{psf-select-example}
    \end{figure}

- paper doesn't mention how they compute the latent image
- fast deconvolution in frequency domain results in ringing artifacts in restored image -> this would affect candidate selection -> use more accurate spatial IRLS-method which is very slow
- if :math:`I^k` is correct should contain salient edges -> compute :math:`\tilde{I^k}`: Gaussian smoothed (reduce noise) and shock filtered (significant edges)

- cross correlation of gradient magnitudes between :math:`I^k` and :math:`\tilde{I^k}`
- only salient edges will not be changed significantly: in blurred images almost all edges will alter through shock filtering and in images with ringing artifacts and other structural problems the edges are ruined too -> correlation value decreases
- example for PSF selection see figure :ref:`psf-select-example`



Blur Removal
++++++++++++

- deblurring of each depth layer

.. math:: :numbered:

    E(I) = \| I \otimes k^d - B \|^2 +  \gamma_f \|\nabla I \|^2

**problem**:

- region boundaries (because dmaps haven't 100% correct boundaries) -> set :math:`\gamma_f` three times larger for pixel with distant to the boundary smaller than kernel size



Second Run
++++++++++

- the deblurred images are used to refine the disparity map
- then run the other steps again