The proposed depth-aware motion deblurring algorithm yields good deblurring
outcomes making it desirable to reproduce its results. As shown in this study
thesis some details of this algorithm could be improved like checking for too
small texture-less regions. Therefore a reference implementation is useful. In
the end the results of the paper can not be achieved by the current version of
the reference implementation. Many difficulties were solved like handling
arbitrary shaped regions in PSF estimation or deconvolution but some
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
