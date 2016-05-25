# Tools

Some small programs that make it possible to use just some code snippets from the original project, like the convolution method.

**conv2** - convolves an image with a kernel

**deconv** - deconvolves an image with a kernel (FFT and IRLS method from Levin converted from matlab to C++)

**top-level-deconv** - Runs the depth-aware motion deblurring algorithm only on the top-level regions (without psf-refinement in the mid-level regions). To set any of the parameter of the algorithm please change the code.