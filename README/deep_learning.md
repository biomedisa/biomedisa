## Deep Learning
```python
biomedisa.deeplearning.deep_learning(
    img_data,
    label_data=None,
    val_img_data=None,
    val_label_data=None,
    val_images=None,
    val_labels=None,
    validation_split=0.0,
    val_dice=True,
    train_dice=False,
    average_dice=False,
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
    batch_size=None,
    early_stopping=0,
    balance=False,
    flip_x=False,
    flip_y=False,
    flip_z=False,
    rotate=0.0,
    swapaxes=False,
    compression=True,
    ignore='none',
    only='all',
    normalization=True,
    x_scale=256,
    y_scale=256,
    z_scale=256,
    scaling=True,
    crop_data=False,
    cropping_epochs=50,
    save_cropped=False,
    pretrained_model=None,
    fine_tune=False,
    return_probs=False,
    patch_normalization=False,
    z_patch=64,
    y_patch=64,
    x_patch=64,
    acwe=False,
    acwe_alpha=1.0,
    acwe_smooth=1,
    acwe_steps=3,
    clean=None,
    fill=None,
    header_file=None
)
```
#### Parameters:
+ **img_data : array_like or list**

    Array_like or list of array-like objects (each image volume must be three-dimensional).

+ **label_data : array_like or list (training)**

    Array_like or list of fully segmented array-like label data (each label volume must be three dimensional). The background area has the value 0.

+ **val_img_data : array_like or list (optional)**

    Array_like or list of array-like objects for validation (each image volume must be three-dimensional).

+ **val_label_data : array_like or list (optional)**

    Array_like or list of fully segmented array-like label data for validation (each label volume must be three dimensional). The background area has the value 0.

#### Returns:
+ **out : dictionary**

    Dictionary containing array-like objects for the results {'regular', 'cropped_volume', 'probs', 'header'} when available.

#### Other Parameters (use leading `--` for command-line, e.g. `--help`):

+ **help**: Show more information and exit (command-line only).
+ **version**: Show Biomedisa version (command-line only).
+ **val_images PATH**: Location of validation image data (tarball, directory, or file) (command-line only).
+ **val_labels PATH**: Location of validation label data (tarball, directory, or file) (command-line only).
+ **validation_split FLOAT**: Splits your dataset into two parts: a training set and a validation set, e.g., `-vs 0.8` indicates that 80% of your data will be used for training, while the remaining 20% will be used for validation.
+ **val_dice**: Monitor Dice score on validation data (default: True). The Dice score offers a more reliable assessment by measuring the overlap between the segmented regions, whereas the standard accuracy also considers background classification, which can lead to misleading results, especially when dealing with small segments within a much larger volume. Even if half of the segment is mislabeled, the standard accuracy may still yield a remarkably high value. However, calculating the Dice score in Biomedisa is computationally intensive and can be disabled with this variable.
+ **train_dice**: Monitor Dice score on training data (default: False).
+ **average_dice**: Monitor averaged dice score of all labels (default: False).
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
+ **batch_size INT**: Batch size (default: None). If not specified, it will be adjusted to the available GPU memory, with a minimum of 6 and a maximum of 24.
+ **early_stopping INT**: Stop training if there is no improvement after specified number of epochs.
+ **balance**: Balance foreground and background training patches (default: False).
+ **flip_x**: Randomly flip x-axis during training (default: False).
+ **flip_y**: Randomly flip y-axis during training (default: False).
+ **flip_z**: Randomly flip z-axis during training (default: False).
+ **rotate FLOAT**: Randomly rotate during training (default: 0.0).
+ **swapaxes**: Randomly swap two axes during training (default: False).
+ **compression**: Compress segmentation results (default: True).
+ **ignore STR**: Ignore specific label(s), e.g. "2,5,6" (default: none).
+ **only STR**: Segment only specific label(s), e.g. "1,3,5" (default: all).
+ **normalization**: Normalize all 3D image volumes to the same mean and variance (default: True).
+ **x_scale INT**: Images and labels are scaled at x-axis to this size before training (default: 256).
+ **y_scale INT**: Images and labels are scaled at y-axis to this size before training (default: 256).
+ **z_scale INT**: Images and labels are scaled at z-axis to this size before training (default: 256).
+ **scaling**: Resize image and label data to z_scale*y_scale*x_scale voxels (default: True).
+ **crop_data**: Both the training and inference data should be cropped to the region of interest for best performance. As an alternative to manual cropping, you can use Biomedisa's AI-based automatic cropping. After training, auto cropping is automatically applied to your inference data.
+ **cropping_epochs INT**: Epochs the network for auto-cropping is trained (default: 50).
+ **save_cropped**: Save automatically cropped image (default: False).
+ **pretrained_model PATH**: Location of pretrained model (only encoder will be trained if specified) (default: None).
+ **fine_tune**: Fine-tune the entire pretrained model. Choose a smaller learning rate, e.g. 0.0001' (default: False).
+ **return_probs**: Return prediction probabilities for each label (default: False).
+ **patch_normalization**: Scale each patch to mean zero and standard deviation (default: False).
+ **x_patch INT**: X-dimension of patch (default: 64).
+ **y_patch INT**: Y-dimension of patch (default: 64).
+ **z_patch INT**: Z-dimension of patch (default: 64).
+ **acwe**: Post-processing with active contour (default: False).
+ **acwe_alpha FLOAT**: Pushing force of active contour (default: 1.0).
+ **acwe_smooth INT**: Smoothing steps of active contour (default: 1).
+ **acwe_steps INT**: Iterations of active contour (default: 3).
+ **clean FLOAT**: Remove outliers, e.g. 0.5 means that objects smaller than 50 percent of the size of the largest object will be removed (default: None).
+ **fill FLOAT**: Fill holes, e.g. 0.5 means that all holes smaller than 50 percent of the entire label will be filled (default: None).
+ **header_file STR**: Location of header file, transfers header information to result (default: None).

#### Pass AMIRA/AVIZO header from image data to result
Label header information from AMIRA/AVIZO training files are automatically preserved. In addition, you can pass image header information, e.g. to preserve information about voxel size. 
```python
from biomedisa.features.biomedisa_helper import load_data, save_data
from biomedisa.deeplearning import deep_learning

# load image data
img, img_header = load_data('image_data.am')

# automatic segmentation
results = deep_learning(img, predict=True, img_header=img_header,
        path_to_model='my_model.h5')

# save result
save_data('segmentation.am', results['regular'],
        header=results['header'])
```

#### Python example NRRD
Label header information different from AMIRA/AVIZO is not automatically transferred. However, you can specify a header file to provide header information for the result. Additionally, you can pass image header information to your result, e.g. to preserve information about voxel size.
```python
from biomedisa.features.biomedisa_helper import load_data, save_data
from biomedisa.deeplearning import deep_learning

# load image data
img, img_header = load_data('image_data.nrrd')

# automatic segmentation
results = deep_learning(img, predict=True, img_header=img_header,
        path_to_model='my_model.h5', header_file='reference_label.nrrd')

# save result
save_data('segmentation.nrrd', results['regular'],
        header=results['header'])
```
Using command line it would be
```
python -m biomedisa.deeplearning 'image_data.nrrd' 'my_model.h5' --header_file='reference_label.nrrd' --predict
```
