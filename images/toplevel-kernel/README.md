# sample PSF kernels

This kernels were created with the [two-phase kernel estimation .exe][Xu10].

The edge tapered top-level regions (region max set to 3) were saved and used as input for deblurring this region. The kernel was saved and resized because of the different kernel sizes:

```bash
# extent or crop the image to the given size (choose reasonable)
convert kernel0.png -background black -gravity center -extent 55x55 kernel0.png
```


[Xu10]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.170.6990&rep=rep1&type=pdf