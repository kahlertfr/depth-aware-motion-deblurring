# Tools

Some small programs that make it possible to use just some code snippets from the original project, like the convolution method.


## generally useful 

**conv2** - convolves an image with a kernel

```bash
conv2 <image> <kernel>
```


**deconv** - deconvolves an image with a kernel (FFT and IRLS method from Levin converted from matlab to C++). A mask for the IRLS method can be specified.

```bash
deconv <image> <kernel> [<mask>]
```


**shock-filter** - Shock filters an image with the coherence filter.

```bash
shock-filter <image>
```

**disparity** - disparity estimation with SGBM and graph-cut

```bash
disparity <left view> <right view>
```



## depth-aware deblurring steps

**top-level-deconv** - Runs the depth-aware motion deblurring algorithm only with the top-level region PSFs (without psf-refinement in the mid-level regions). To set any of the parameter of the algorithm please change the code.

```bash
top-level-deconv <left> <right>
```


**psf-selection** - Runs only the PSF selection part of the algorithm: selects the best kernel for deblurring from given kernel candidates.

```bash
psf-selection <image> <psf1> <psf2> <psf3> [<mask>]
```
