## Deep Learning
```python
demo.deep_learning(
    img_data,
    label_data=None,
    val_img_data=None,
    val_label_data=None,
    val_images=None,
    val_labels=None,
    validation_split=0.0,
    val_tf=False,
    path_to_model=None,
    predict=False,
    train=False,
    network_filters='32-64-128-256-512',
    resnet=False,
    epochs=100,
    learning_rate=0.01,
    stride_size=32,
    validation_stride_size=32,
    validation_freq=1,
    batch_size=24,
    early_stopping=0,
    balance=False,
    flip_x=False,
    flip_y=False,
    flip_z=False,
    rotate=0.0,
    no_compression=False,
    create_slices=False,
    ignore='none',
    only='all',
    no_normalization=False,
    clean=None,
    fill=None,
    x_scale=256,
    y_scale=256,
    z_scale=256,
    no_scaling=False,
    crop_data=False,
    cropping_epochs=50,
    save_cropped=False
)
```
#### Parameters (Python only):
+ **img_data : list**

    List of array-like objects (each image volume must be three-dimensional).

+ **label_data : list (training only)**

    List of fully segmented array-like label data (each label volume must be three dimensional). The background area has the value 0.

+ **val_img_data : list (optional)**

    List of array-like objects for validation (each image volume must be three-dimensional).

+ **val_label_data : list (optional)**

    List of fully segmented array-like label data for validation (each label volume must be three dimensional). The background area has the value 0.

#### Returns (prediction only):
+ **out : dictionary**

    Dictionary containing array-like objects for the results {'regular', 'cleaned', 'filled', 'cleaned_filled', 'cropped_volume'} when available.

#### Other Parameters (use leading `--` for command-line, e.g. `--help`):

+ **help**: Show more information and exit (command-line only).
+ **version**: Show Biomedisa version (command-line only).
+ **val_images PATH**: Path to directory with validation images (command-line only).
+ **val_labels PATH**: Path to directory with validation labels (command-line only).
+ **validation_split FLOAT**: For example, split your data into 80% training data and 20% validation data with `-vs 0.8`.
+ **val_tf**: Use standard pixelwise accuracy provided by TensorFlow (default: False). When evaluating accuracy, Biomedisa relies on the Dice score rather than the standard accuracy. The Dice score offers a more reliable assessment by measuring the overlap between the segmented regions, whereas the standard accuracy also considers background classification, which can lead to misleading results, especially when dealing with small segments within a much larger volume. Even if half of the segment is mislabeled, the standard accuracy may still yield a remarkably high value. However, if you still prefer to use the standard accuracy, you can enable it by using this option.
+ **path_to_model PATH**: Path to model.
+ **predict**: Automatic/predict segmentation.
+ **train**: Train a neural network.
+ **network_filters STR**: Number of filters per layer up to the deepest, e.g. "32-64-128-256-512" (default: "32-64-128-256-512").
+ **resnet**: Use U-resnet instead of standard U-net (default: False).
+ **epochs INT**: Number of epochs trained (default: 100).
+ **learning_rate FLOAT**: Learning rate (default: 0.01).
+ **stride_size [1-64]**: Stride size for patches (default: 32).
+ **validation_stride_size [1-64]**: Stride size for validation patches (default: 32).
+ **validation_freq INT**: Epochs performed before validation (default: 1).
+ **batch_size INT**: Batch size (default: 24). If you have a memory error, try reducing to 6, for example.
+ **early_stopping INT**: Stop training if there is no improvement after specified number of epochs.
+ **balance**: Balance foreground and background training patches (default: False).
+ **flip_x**: Randomly flip x-axis during training (default: False).
+ **flip_y**: Randomly flip y-axis during training (default: False).
+ **flip_z**: Randomly flip z-axis during training (default: False).
+ **rotate FLOAT**: Randomly rotate during training (default: 0.0).
+ **no_compression**: Disable compression of segmentation results (default: False).
+ **create_slices**: Create slices of segmentation results (default: False).
+ **ignore STR**: Ignore specific label(s), e.g. "2,5,6" (default: none).
+ **only STR**: Segment only specific label(s), e.g. "1,3,5" (default: all).
+ **no_normalization**: Disable image normalization (default: False).
+ **clean FLOAT**: Remove outliers, e.g. 0.5 means that objects smaller than 50 percent of the size of the largest object will be removed (default: None).
+ **fill FLOAT**: Fill holes, e.g. 0.5 means that all holes smaller than 50 percent of the entire label will be filled (default: None).
+ **x_scale INT**: Images and labels are scaled at x-axis to this size before training (default: 256).
+ **y_scale INT**: Images and labels are scaled at y-axis to this size before training (default: 256).
+ **z_scale INT**: Images and labels are scaled at z-axis to this size before training (default: 256).
+ **no_scaling**: Do not resize image and label data (default: False).
+ **crop_data**: Both the training and inference data should be cropped to the region of interest for best performance. As an alternative to manual cropping, you can use Biomedisa's AI-based automatic cropping. After training, auto cropping is automatically applied to your inference data.
+ **save_cropped**: Save cropped image (default: False).
