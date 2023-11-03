## Smart Interpolation
```python
biomedisa_features.biomedisa_interpolation.smart_interpolation(
    data,
    labelData,
    nbrw=10,
    sorw=4000,
    acwe=False,
    acwe_alpha=1.0,
    acwe_smooth=1,
    acwe_steps=3,
    no_compression=False,
    allaxis=False,
    denoise=False,
    uncertainty=False,
    create_slices=False,
    ignore='none',
    only='all',
    smooth=0,
    clean=None,
    fill=None,
    platform=None
)
```
#### Parameters:
+ **data : array_like**

    Image data (must be three-dimensional).

+ **labelData : array_like**

    Pre-segmented slices (must be three dimensional). The non-segmented area has the value 0.

#### Returns:
+ **out : dictionary**

    Dictionary containing array-like objects for the results {'regular', 'smooth', 'uncertainty', 'cleaned', 'filled', 'cleaned_filled', 'acwe'} when available.

#### Other Parameters (use leading `--` for command-line, e.g. `--help`):

+ **help**: Show more information and exit (command-line only).
+ **version**: Show Biomedisa version (command-line only).
+ **nbrw INT**: Number of random walks starting at each pre-segmented pixel (default: 10).
+ **sorw INT**: Steps of a random walk (default: 4000).
+ **acwe**: Post-processing result with active contour (default: False).
+ **acwe_alpha FLOAT**: Pushing force of active contour (default: 1.0).
+ **acwe_smooth INT**: Smoothing of active contour (default: 1).
+ **acwe_steps INT**: Iterations of active contour (default: 3).
+ **no_compression**: Disable compression of segmentation results (default: False).
+ **allaxis**: If pre-segmentation is not exlusively in xy-plane (default: False).
+ **denoise**: Smooth/denoise image data before processing (default: False).
+ **uncertainty**: Return uncertainty of segmentation result (default: False).
+ **create_slices**: Create slices of segmentation results (default: False).
+ **ignore STR**: Ignore specific label(s), e.g. "2,5,6" (default: none).
+ **only STR**: Segment only specific label(s), e.g. "1,3,5" (default: all).
+ **smooth INT**: Number of smoothing iterations for segmentation result (default: 0).
+ **clean FLOAT**: Remove outliers, e.g. 0.5 means that objects smaller than 50 percent of the size of the largest object will be removed (default: None).
+ **fill FLOAT**: Fill holes, e.g. 0.5 means that all holes smaller than 50 percent of the entire label will be filled (default: None).
+ **platform STR**: One of "cuda", "opencl_NVIDIA_GPU", "opencl_Intel_CPU" (default: None).

#### Multi-GPU (e.g. 4 GPUs)
```
# Ubuntu
mpiexec -np 4 python3 biomedisa_interpolation.py ~/Downloads/NMB_F2875.tif ~/Downloads/labels.NMB_F2875.tif

# Windows
mpiexec -np 4 python -u biomedisa_interpolation.py Downloads\NMB_F2875.tif Downloads\labels.NMB_F2875.tif
```

#### Memory error
If memory errors (either GPU or host memory) occur, you can start the segmentation as follows:
```
python3 git/biomedisa/biomedisa_features/split_volume.py 'path_to_image' 'path_to_labels' -np 4 -sz 2 -sy 2 -sx 2
```
Where `-n` is the number of GPUs and each axis (`x`,`y` and `z`) is divided into two overlapping parts. The volume is thus divided into `2*2*2=8` subvolumes. These are segmented separately and then reassembled.
