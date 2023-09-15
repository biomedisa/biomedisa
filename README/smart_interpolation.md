## Smart Interpolation
```python
demo.biomedisa_interpolation.smart_interpolation(
    data, labelData,
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
    no_compression=False):
```
#### Parameters (Python only):
+ data : array_like
    Image data (must be three-dimensional).

+ labelData : array_like
    Pre-segmented slices (must be three dimensional). The non-segmented area has the value 0.

#### Returns:
+ out : dictionary
    Dictionary containing array-like objects for the results {'regular', 'smooth', 'uncertainty', 'cleaned', 'filled', 'cleaned_filled', 'acwe'} when available.

#### Other Parameters (abbreviations for command-line only):

`--help` or `-h`: show more information and exit

`--version` or `-v`: Biomedisa version

`--nbrw INT`: number of random walks starting at each pre-segmented pixel (default: 10)

`--sorw INT`: steps of a random walk (default: 4000)

`--acwe`: post-processing with active contour (default: False)

`--acwe_alpha FLOAT`: pushing force of active contour (default: 1.0)

`--acwe_smooth INT`: smoothing of active contour (default: 1)

`--acwe_steps INT`: iterations of active contour (default: 3)

`--no_compression` or `-nc`: disable compression of segmentation results (default: False)

`--allaxis` or `-allx`: if pre-segmentation is not exlusively in xy-plane (default: False)

`--denoise` or `-d`: smooth/denoise image data before processing (default: False)

`--uncertainty` or `-u`: return uncertainty of segmentation result (default: False)

`--create_slices` or `-cs`: create slices of segmentation results (default: False)

`--ignore STR`: ignore specific label(s), e.g. "2,5,6" (default: none)

`--only STR`: segment only specific label(s), e.g. "1,3,5" (default: all)

`--smooth INT` or `-s INT`: number of smoothing iterations for segmentation result (default: 0)

`--clean FLOAT` or `-c FLOAT`: remove outliers, e.g. 0.5 means that objects smaller than 50 percent of the size of the largest object will be removed (default: None)

`--fill FLOAT` or `-f FLOAT`: fill holes, e.g. 0.5 means that all holes smaller than 50 percent of the entire label will be filled (default: None)

`--platform STR` or `-p STR`: one of "cuda", "opencl_NVIDIA_GPU", "opencl_Intel_CPU" (default: None)
