## Smart Interpolation
```python
demo.biomedisa_interpolation.smart_interpolation(
    data,
    labelData,
    nbrw=10,
    sorw=4000,
    acwe=False,
    acwe_alpha=1.0,
    acwe_smooth=1,
    acwe_steps=3,
    denoise=False,
    uncertainty=False,
    create_slices=False,
    platform=None,
    allaxis=False,
    ignore='none',
    only='all',
    clean=None,
    fill=None,
    smooth=0,
    no_compression=False
)
```
#### Parameters (Python only):
+ **data : array_like**

    Image data (must be three-dimensional).

+ **labelData : array_like**

    Pre-segmented slices (must be three dimensional). The non-segmented area has the value 0.

#### Returns:
+ **out : dictionary**

    Dictionary containing array-like objects for the results {'regular', 'smooth', 'uncertainty', 'cleaned', 'filled', 'cleaned_filled', 'acwe'} when available.

#### Other Parameters (abbreviations for command-line only):

+ **--help or -h** 

    Show more information and exit (command-line only).

+ **--version or -v**

    Show Biomedisa version (command-line only).

+ **--nbrw INT**

    Number of random walks starting at each pre-segmented pixel (default: 10).

+ **--sorw INT**

    Steps of a random walk (default: 4000).

+ **--acwe**

    Post-processing result with active contour (default: False).

+ **--acwe_alpha FLOAT**

    Pushing force of active contour (default: 1.0).

+ **--acwe_smooth INT**

    Smoothing of active contour (default: 1).

+ **--acwe_steps INT**

    Iterations of active contour (default: 3).

+ **--no_compression or -nc**

    Disable compression of segmentation results (default: False).

+ **--allaxis or -allx**

    If pre-segmentation is not exlusively in xy-plane (default: False).

+ **--denoise or -d**

    Smooth/denoise image data before processing (default: False).

+ **--uncertainty or -u**

    Return uncertainty of segmentation result (default: False).

+ **--create_slices or -cs**

    Create slices of segmentation results (default: False).

+ **--ignore STR**

    Ignore specific label(s), e.g. "2,5,6" (default: none).

+ **--only STR**

    Segment only specific label(s), e.g. "1,3,5" (default: all).

+ **--smooth INT or -s INT**

    Number of smoothing iterations for segmentation result (default: 0).

+ **--clean FLOAT or -c FLOAT**

    Remove outliers, e.g. 0.5 means that objects smaller than 50 percent of the size of the largest object will be removed (default: None).

+ **--fill FLOAT or -f FLOAT**

    Fill holes, e.g. 0.5 means that all holes smaller than 50 percent of the entire label will be filled (default: None).

+ **--platform STR or -p STR**

    One of "cuda", "opencl_NVIDIA_GPU", "opencl_Intel_CPU" (default: None)
