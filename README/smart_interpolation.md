## Smart Interpolation
```python
biomedisa.interpolation.smart_interpolation(
    data,
    labelData,
    nbrw=10,
    sorw=4000,
    no_compression=False,
    allaxis=False,
    denoise=False,
    uncertainty=False,
    ignore='none',
    only='all',
    smooth=0,
    platform=None,
    return_hits=False,
    acwe=False,
    acwe_alpha=1.0,
    acwe_smooth=1,
    acwe_steps=3,
    clean=None,
    fill=None
)
```
#### Parameters:
+ **data : array_like**

    Image data (must be three-dimensional).

+ **labelData : array_like**

    Pre-segmented slices (must be three dimensional). The non-segmented area has the value 0.

#### Returns:
+ **out : dictionary**

    Dictionary containing array-like objects for the results {'regular', 'smooth', 'uncertainty', 'hits'} when available.

#### Other Parameters (use leading `--` for command-line, e.g. `--help`):

+ **help**: Show more information and exit (command-line only).
+ **version**: Show Biomedisa version (command-line only).
+ **nbrw INT**: Number of random walks starting at each pre-segmented pixel (default: 10).
+ **sorw INT**: Steps of a random walk (default: 4000).
+ **no_compression**: Disable compression of segmentation results (default: False).
+ **allaxis**: If pre-segmentation is not exlusively in xy-plane (default: False).
+ **denoise**: Smooth/denoise image data before processing (default: False).
+ **uncertainty**: Return uncertainty of segmentation result (default: False).
+ **ignore STR**: Ignore specific label(s), e.g. "2,5,6" (default: none).
+ **only STR**: Segment only specific label(s), e.g. "1,3,5" (default: all).
+ **smooth INT**: Number of smoothing iterations for segmentation result (default: 0).
+ **platform STR**: One of "cuda", "opencl_NVIDIA_GPU", "opencl_Intel_CPU" (default: None).
+ **return_hits**: Return hits from each label. Only works for small image data (default: False).
+ **acwe**: Post-processing with active contour (default: False).
+ **acwe_alpha FLOAT**: Pushing force of active contour (default: 1.0).
+ **acwe_smooth INT**: Smoothing steps of active contour (default: 1).
+ **acwe_steps INT**: Iterations of active contour (default: 3).
+ **clean FLOAT**: Remove outliers, e.g. 0.5 means that objects smaller than 50 percent of the size of the largest object will be removed (default: None).
+ **fill FLOAT**: Fill holes, e.g. 0.5 means that all holes smaller than 50 percent of the entire label will be filled (default: None).

#### Multi-GPU (e.g. 4 GPUs)
```
mpiexec -np 4 python3 -m biomedisa.interpolation Downloads\NMB_F2875.tif Downloads\labels.NMB_F2875.tif
```

#### If you encounter GPU or host memory issues, you can split your volume into smaller segments and merge the results. For instance, you could use 8 sub-volumes
```
python -m biomedisa.features.split_volume Downloads\NMB_F2875.tif Downloads\labels.NMB_F2875.tif --split_x=2 --split_y=2 --split_z=2
```

