# Tools

Some small programs that make it possible to use just some code snippets from the original project, like the convolution method.

## generally useful 

**conv2** - convolves an image with a kernel

**deconv** - deconvolves an image with a kernel (FFT and IRLS method from Levin converted from matlab to C++)

**shock-filter** - Shock filters an image with the coherence filter.


## depth-aware deblurring steps

**top-level-deconv** - Runs the depth-aware motion deblurring algorithm only with the top-level region PSFs (without psf-refinement in the mid-level regions). To set any of the parameter of the algorithm please change the code.

**psf-selection** - Runs only the PSF selection part of the algorithm: selects the best kernel for deblurring from given kernel candidates.