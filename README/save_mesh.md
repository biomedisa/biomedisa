## Save Mesh
```python
biomedisa.mesh.save_mesh(
    path_to_result,
    labels,
    x_res=1.0,
    y_res=1.0,
    z_res=1.0,
    poly_reduction=0.9,
    smoothing_iterations=15
)
```
#### Parameters:
+ **path_to_result : PATH**

    Path to the location of the generated mesh file.

+ **labels : array_like**

    Label data (must be three dimensional). The non-segmented area must have the value 0.

#### Other Parameters (use leading `--` for command-line, e.g. `--help`):

+ **help**: Show more information and exit (command-line only).
+ **x_res FLOAT**: Voxel spacing/resolution x-axis (default: None).
+ **y_res FLOAT**: Voxel spacing/resolution y-axis (default: None).
+ **z_res FLOAT**: Voxel spacing/resolution z-axis (default: None).
+ **poly_reduction FLOAT**: Reduce number of polygons by this factor (default: 0.9).
+ **smoothing_iterations INT**: Iteration steps for smoothing (default: 15).

