# Study Thesis

My study thesis (Belegarbeit) in computer vision based on the paper ["Depth-Aware Motion Deblurring"][Xu12] of Xu and Jia.

They use spatially-varying point spread functions to deblur the image depending on the depth layer. The results can be found here: [website][Xu-website].


## Building

The project structure is modular. It contains the depth-aware motion deblurring algorithm and the two-phase kernel estimation algorithm which can be used independently. You can build all components from the toplevel following the next steps.


### Requirements

- [OpenCV 3.0](http://opencv.org/). [Installation guide][OpenCV-install]


### CMake configuration

[CMake](http://cmake.org/) is the used build tool. Use an additional build folder to have clean source folders.

```bash
# create a build directory (all CMake Files will be saved there)
mkdir build
cd build

# Create a build configuration
# CMAKE_BUILD_TYPE= Release for disabling output (doesn't show/save images ...)
cmake -D CMAKE_BUILD_TYPE=Release ..

# use make for building all make targets or specify the needed target
make
```


### Make targets

#### motion-deblurring

This is the main algorithm.

```bash
make motion-deblurring

# Executable can be found in build/bin
bin/motion-deblurring ../images/mouse-left.jpg ../images/mouse-right.jpg
```

#### two-phase-kernel

This part of the Depth-Aware Motion Deblurring Algorithm can be used completly independent of the whole algorithm

```bash
make two-phase-kernel

# Executable can be found in build/bin
bin/two-phase-kernel../images/mouse-left.jpg
```



## Literature on Motion Deblurring

### Books

- A. N. Rajagopalan, Rama Chellappa - Motion Deblurring: Algorithms and Systems (ISBN 9781107044364)



### Paper (sorted by relevance)

- L. Xu, J. Jia. [Depth-Aware Motion Deblurring, IEEE 2012][Xu12]



#### Main Referemces of "Depth-Aware Motion Deblurring"

- L. Xu, J. Jia. [Two-phase Kernel Estimation for Robust Motion Deblurring, ECCV 2010][Xu10]
    - technique used for top level PSF computation
    - strong edges not always good for kernel estimation
- O. Whyte, J. Sivic, A. Zisserman, and J. Ponce.  [Non-uniform Deblurring for Shaken Images, CVPR 2010][Whyte10]
    - models 3D rotation of camera
    - single image and sharp + noisy image approach
- N. Joshi, S.B. Kang, L. Zitnick, and R. Szeliski. [Image Deblurring with Inertial Measurement Sensors. ACM SIGGRAPH 2010][Joshi10]
    - hardware attachment for single image deblurring
- A. Gupta, N. Joshi, L. Zitnick, M. Cohen, and B. Curless. [Single Image Deblurring Using Motion Density Functions, ECCV 2010][Gupta10]
    - using spatially invariant deconvolution methods in a local and robust way
- S. Cho and S. Lee. [Fast motion deblurring. ACM Trans. Graph., 28(5), 2009][Cho09]
    - technique used for iterative PSF computation
    - iterative single image approach
    - novel predictive step (strong edge) and kernel estimation based on derivatives



#### Additional Papers

- H. Qiu. [State-of-the-Art Image Motion Deblurring Technique][Qiu]
    - conclusion on several single image and stereo image, blind and non-blind deblurring algorithms
- B. Kalaiyarasi, S. Kalpana. [Blind Deconvolution of Camera Motioned Picture using Depth Map][Kalaiyarashi2012]
    - camera shake and large depth range scene leads to non-uniform blur
    - single image with depth map
- J. Jia. [Single Image Motion Deblurring Using Transparency, IEEE 2007][Jia2007]
    - can handle camera motion blur and object motion blur
    - investigates relationship between object boundary transparency and image motion blur
- T. Kobayashi, F. Sakaue, & J. Sato. [Depth and Arbitrary Motion Deblurring Using Integrated PSF, ECCV 2014][Kobayashi14]
    - motion deblurring and all-in-focus imaging can be achieved simultaneously
    - motion blur caused by arbitrary multiple motions can be recovered
- Z. Hu, L. Xu, M. Yang [Joint Depth Estimation and Camera Shake Removal from Single Blurry Image, CVPR2014][Hu2014]
- H. Hirschmüller [Stereo Preocessing by Semi-Global Matching and Mutual Information, IEEE 2007][Hirschmüller2007]
    - description of SGBM-Algorithm used for disparity estimation



[Cho09]: http://rosaec.snu.ac.kr/publish/2009/ID/ChLe-SIGGRAPH-2009.pdf
[Gupta10]: http://grail.cs.washington.edu/projects/mdf_deblurring/gupta_mdf_deblurring.pdf
[Hirschmüller2007]: http://core.ac.uk/download/pdf/11134866.pdf
[Hu2014]: https://eng.ucmerced.edu/people/zhu/CVPR14_deblurdepth.pdf
[Jia2007]: http://www.cse.cuhk.edu.hk/~leojia/all_final_papers/motion_deblur_cvpr07.pdf
[Joshi10]: http://202.114.89.42/resource/pdf/7570.pdf
[Kalaiyarashi2012]: http://ijarece.org/wp-content/uploads/2015/02/IJARECE-VOL-4-ISSUE-2-142-148.pdf
[Kobayashi14]: http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/ECCV_2014/workshops/w14/Kobayashi-et-al-LF4CV2014.pdf
[Qiu]: http://iwct.sjtu.edu.cn/Personal/xwang8/research/hang/State-of-the-Art%20Image%20Motion%20Deblurring.pdf
[Whyte10]: http://www.di.ens.fr/willow/pdfs/cvpr10d.pdf
[Xu10]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.170.6990&rep=rep1&type=pdf
[Xu12]: http://www.cse.cuhk.edu.hk/leojia/papers/depth_deblur_iccp12.pdf

[Xu-website]: https://appsrv.cse.cuhk.edu.hk/~leojia/projects/nonuniform_deblur/index.html

[OpenCV-install]: http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html#table-of-content-introduction
