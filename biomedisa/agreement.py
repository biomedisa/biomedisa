#!/usr/bin/python3
import os
import numpy as np
from biomedisa.features.biomedisa_helper import unique
from biomedisa.matching import assign_labels, remove_small_particles2
from platformdirs import user_data_dir
from pathlib import Path
from tqdm import tqdm
import requests
import time

biomedisa_dir = Path(user_data_dir("biomedisa"))
biomedisa_dir.mkdir(parents=True, exist_ok=True)

models_dir = biomedisa_dir / "models"
models_dir.mkdir(exist_ok=True)

print("Saving models in:", models_dir)


def match_particles_old(result1, result2):
    # get labels
    labels1, sizes1 = unique(result1, return_counts=True)
    labels2, sizes2 = unique(result2, return_counts=True)

    # assign reference value to the largest overlapping label
    print("Total particles:", len(labels1)-1, len(labels2)-1)
    labels_matrix = np.zeros((np.amax(labels1)+1, np.amax(labels2)+1), np.uint32)
    labels_matrix = assign_labels(result1, result2, labels_matrix)

    label_vals = np.zeros(np.amax(labels1)+1, dtype=np.uint8)

    for i1, val1 in enumerate(labels1[1:]):
        val2 = np.argmax(labels_matrix[val1])
        i2 = np.argwhere(labels2==val2)[0][0]
        dice = 2*labels_matrix[val1,val2] / float(sizes1[i1+1]+sizes2[i2]+1e-6)
        if dice > 0.9:
            label_vals[val1] = 1

    print("Agreed particles:", np.sum(label_vals))
    result1 = remove_small_particles2(result1, label_vals)
    return result1


def match_particles(results, dice_threshold=0.9, min_agreement=None):
    """
    Parameters
    ----------
    results : list of ndarray
        List of labeled instance segmentations.
        The first segmentation is used as the reference.

    dice_threshold : float
        Minimum Dice score required for a particle match.

    min_agreement : int or None
        Number of segmentations (including the reference) that must agree.
        None -> require agreement from all segmentations.

    Returns
    -------
    agreed_labels : ndarray
        Reference segmentation with disagreed particles removed.
    """

    reference = results[0]
    labels_ref, sizes_ref = unique(reference, return_counts=True)

    # votes for each reference particle
    votes = np.ones(np.amax(labels_ref) + 1, dtype=np.uint16)  # reference votes for itself

    for result in results[1:]:
        labels, sizes = unique(result, return_counts=True)
        labels_matrix = np.zeros((np.amax(labels_ref)+1, np.amax(labels)+1), np.uint32)
        labels_matrix = assign_labels(reference, result, labels_matrix)

        for i_ref, label_ref in enumerate(labels_ref[1:]):

            label = np.argmax(labels_matrix[label_ref])
            if label == 0:
                continue

            idx = np.argwhere(labels == label)
            if len(idx) == 0:
                continue
            idx = idx[0,0]

            overlap = labels_matrix[label_ref, label]
            dice = 2.0 * overlap / (sizes_ref[i_ref+1] + sizes[idx] + 1e-6)
            if dice > dice_threshold:
                votes[label_ref] += 1

    if min_agreement is None:
        min_agreement = len(results)

    label_vals = np.zeros(np.amax(labels_ref)+1, np.uint8)
    label_vals[votes >= min_agreement] = 1

    print(f"Agreed particles: {np.sum(label_vals)} / {len(labels_ref)-1}")

    return remove_small_particles2(reference, label_vals)


def download_model(model="sam"):
    if model=="sam":
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        output_file = os.path.join(models_dir, "sam_vit_l_0b3195.pth")
    else:
        url = "https://biomedisa.info/media/Quartz/model_svl_step=2.h5"
        output_file = os.path.join(models_dir, "model_svl_step=2.h5")

    if not os.path.exists(output_file):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))

            with open(output_file, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True) as bar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        bar.update(len(chunk))
        print(f"Saved to {model_file}")
    return model_file


def agreement_finetune(image, mask, path_to_model=None, labels=None, dice_threshold=0.9, min_agreement=None):
    from biomedisa.deeplearning import deep_learning

    if labels is None:

        # get models
        models = []
        if path_to_model is None:
            models.append(download_model("biomedisa"))
            models.append(download_model("sam"))
        elif isinstance(path_to_model, str):
            models.append(path_to_model)
            models.append(download_model("sam"))
        elif isinstance(path_to_model, list):
            if len(path_to_model) < 2:
                raise ValueError("path_to_model must contain at least two model paths")
            models = path_to_model
        else:
            raise TypeError("path_to_model must be None, a string, or a list of at least two paths")

        # predict results
        labels = [
            deep_learning(
                image,
                mask_data=mask,
                predict=True,
                path_to_model=model,
                batch_size=512
            )["regular"]
            for model in models
        ]

    elif len(labels) < 2:
        raise ValueError("Need at least two label volumes.")

    # match particles
    labels = match_particles(labels, dice_threshold, min_agreement)

    # new model path
    if models and models[0].lower().endswith(".h5"):
        pretrained = True
        model_path = models[0].replace(".h5", "_finetuned.h5")
    else:
        pretrained = False
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S")
        model_path = f"biomedisa_{current_time}.h5"
    print("New model path:" , model_path)

    # validation split
    val_split = 0.8
    val_split = int(val_split*image.shape[0])
    val_img = image[val_split:].copy()
    image = image[:val_split].copy()
    val_label = labels[val_split:].copy()
    labels = labels[:val_split].copy()

    # finetune model
    deep_learning(image, labels, train=True,
        path_to_model = model_path,
        pretrained_model = (models[0] if pretrained else None),
        scaling=False, val_dice=False, patch_normalization=True,
        flip_x=True, flip_y=True, flip_z=True, swapaxes=True,
        val_img_data=val_img, val_label_data=val_label,
        x_patch=16, y_patch=16, z_patch=16, batch_size=48,
        stride_size=8, validation_stride_size=16, separation=True)

    # predict updated result
    results = deep_learning(image, mask_data=mask, predict=True, path_to_model=model_path, batch_size=512)

    return results['regular']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Biomedisa agreement fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "image_path",
        help="Path to the image volume"
    )

    parser.add_argument(
        "mask_path",
        help="Path to the binary mask"
    )

    parser.add_argument(
        "-m", "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=("One or more model paths (.h5, .pth, .pt). "
              "If omitted, the default Biomedisa and SAM models are used.")
    )

    parser.add_argument(
        "-l", "--labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help=("Optional precomputed label volumes. "
              "If provided, predictions are skipped.")
    )

    parser.add_argument(
        "--dice",
        type=float,
        default=0.9,
        help="Minimum Dice score for agreement."
    )

    parser.add_argument(
        "--min_agreement",
        type=int,
        default=None,
        help=("Minimum number of segmentations that must agree. "
              "Default: all models.")
    )

    args = parser.parse_args()

    agreement_finetune(
        image=args.image_path,
        mask=args.mask_path,
        path_to_model=args.models,
        labels=args.labels,
        dice_threshold=args.dice,
        min_agreement=args.min_agreement,
    )

