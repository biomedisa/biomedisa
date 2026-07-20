import sys, os
from tifffile import imread, imwrite
from biomedisa.features.biomedisa_helper import *
from biomedisa.interpolation import smart_interpolation
import numpy as np
#from asd import ASSD

def Dice(ground_truth, result):
    dice = 2 * np.logical_and(ground_truth==result, (ground_truth+result)>0).sum() / \
    float((ground_truth>0).sum() + (result>0).sum())
    return dice

def average_surface_distance(a, b):
    number_of_elements = 0
    distances = 0
    hausdorff = 0
    for label in np.unique(a)[1:]:
        d, n, h = ASSD(a, b, label)
        number_of_elements += n
        distances += d
        hausdorff = max(h, hausdorff)
    assd = distances / float(number_of_elements)
    return assd, hausdorff

def read_indices(volData):
    i = []
    for k, slc in enumerate(volData[:]):
        if np.amax(slc) != 0:
            i.append(k)
    return i

if __name__ == "__main__":

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #=======================================================================================
    # teeth
    #=======================================================================================

    dice, assd_score, hausdorff = [], [], []

    for sample in range(1,6):
        for side in ['left','right']:

            # load image
            if rank==0:
                img = imread(f"teeth/1000340{sample}_000_001_0000.transformed_M1_{side}.tif")

            # cross validation
            for cv in [0,1]:

                # load labels
                if rank==0:
                    labels = imread(f"teeth/1000340{sample}_000_001_0000.transformed_M1_{side}_labels.tif")
                    ref = labels.copy()

                    # cross validation
                    indices = read_indices(labels)
                    for i in indices[cv::2]:
                        labels[i] = 0
                else:
                    img = None
                    labels = None
                comm.Barrier()

                # smart interpolation with optional smoothing result
                results = smart_interpolation(img, labels, smooth=0)

                # get result
                if rank==0:
                    result = results['regular']

                    # reference
                    ref[labels>0] = 0
                    for k in range(ref.shape[0]):
                        if not np.any(ref[k]):
                            result[k] = 0

                    #ac, h = average_surface_distance(a, b)
                    #assd_score.append(ac)
                    #hausdorff.append(h)

                    d = Dice(ref, result)
                    dice.append(d)

    if rank==0:
        print('Dice:', np.mean(dice), np.std(dice))
        #print('ASSD:', np.mean(assd_score), np.std(assd_score))
        #print('Hausdorff:', np.mean(hausdorff), np.std(hausdorff))

    #=======================================================================================
    # cross-validation
    #=======================================================================================

    # images
    images = [
             "cockroach/cockroach_image.tif",
             "dragon_claw/drogon_claw_01_slices.tif",
             "medaka/fish_rebuild_TEb5LYL.tif",
             "wasp-mineralized/NMB_F2875.tif",
             "wasp-amber/megaspilidae_1805-6__slices.filled.tif",
             "ant/ant_slices.tif"
             ]

    # labels
    labels = [
             "cockroach/cockroach_labels_corr_PL.tif",
             "dragon_claw/drogon_claw_01_labels.tif",
             "medaka/skeleton_corr3_TvdK.labels.tif",
             "wasp-mineralized/labels.NMB_F2875.tif",
             "wasp-amber/megaspilidae_1805-6_labels_corr_PL.tif",
             "ant/myrmecia_head_labels_corr.tif"
             ]

    for i, path_to_labels in enumerate(labels):

        # load image
        if rank==0:
            img = imread(images[i])

        dice = []
        for cv in [0,1]:

            # load labels
            if rank==0:
                labels = imread(path_to_labels)
                ref = labels.copy()

                # cross validation
                indices = read_indices(labels)
                for k in indices[cv::2]:
                    labels[k] = 0
            else:
                img = None
                labels = None
            comm.Barrier()

            # smart interpolation with optional smoothing result
            results = smart_interpolation(img, labels, smooth=0)

            # get result
            if rank==0:
                result = results['regular']

                # reference
                ref[labels>0] = 0
                for k in range(ref.shape[0]):
                    if not np.any(ref[k]):
                        result[k] = 0

                d = Dice(ref, result)
                dice.append(d)

        if rank==0:
            print(os.path.basename(path_to_labels), 'Dice:', np.mean(dice), dice)

    #=======================================================================================
    # trigonopterus
    #=======================================================================================

    # labels
    label_paths = ["trigonopterus/labels/labels_smart_neu.tif",
              "trigonopterus/labels/labels_10.tif",
              "trigonopterus/labels/labels_20.tif",
              "trigonopterus/labels/labels_40.tif",
              "trigonopterus/labels/labels_80.tif"
                ]

    # load data
    if rank==0:
        reference = imread("trigonopterus/labels/labels_5.tif")
        img = imread("trigonopterus/CT_0189_Trigonopterus_sp.filled.tif")
    else:
        img = None
        labels = None

    for path_to_labels in label_paths:

        # load labels
        if rank==0:
            labels = imread(path_to_labels)
        comm.Barrier()

        # smart interpolation
        results = smart_interpolation(img, labels, smooth=0)

        # get result
        if rank==0:
            result = results['regular']

            # build reference
            ref = reference.copy()
            ref[labels>0] = 0
            for k in range(ref.shape[0]):
                if not np.any(ref[k]):
                    result[k] = 0

            # calculate dice score
            dice_score = Dice(ref, result)
            print(os.path.basename(path_to_labels), 'Dice:', dice_score)

    #=======================================================================================
    # heart
    #=======================================================================================

    d, a_acc, hausdorff = [], [], []
    for patient in range(10):

        # load image
        img = load_data(f"heart/training_axial_crop_pat{patient}.nii.gz")[0]

        # load ground truth
        gt = load_data(f"heart/training_axial_crop_pat{patient}-label.nii.gz")[0]

        # cross validation
        for k in [0,10]:

            # build pre-segmentation
            labels = np.zeros_like(gt)
            labels[k::20] = gt[k::20]

            # smart interpolation
            results = smart_interpolation(img, labels, smooth=0)
            result = results['regular']

            # reference slices
            ref = np.zeros_like(gt)
            ref[10-k::20] = gt[10-k::20]

            # reference
            for k in range(ref.shape[0]):
                if not np.any(ref[k]):
                    result[k] = 0

            # ASSD
            #ass, h = average_surface_distance(ref,result)
            #a_acc.append(ass)
            #hausdorff.append(h)

            # calculate dice score
            d.append(Dice(ref,result))

    print('Dice:', np.mean(d), np.std(d))
    #print('ASSD:', np.mean(a_acc), np.std(a_acc))
    #print('Hausdorff:', np.mean(hausdorff), np.std(hausdorff))

