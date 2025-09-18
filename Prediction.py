"""
This script is based on SynthSeg's prediction pipeline and has been modified for DRIPS.

Please cite BOTH if you use this code:
- SynthSeg (Benjamin Billot et al.): https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib
- DRIPS repository: https://github.com/LunaBitar1998/DRIPS-Domain-Randomization-Image-based-PVS-Segmentation

Copyright 2020 Benjamin Billot
Modifications © 2025 Luna Bitar

License: Apache License 2.0

"""



# Usage:
# python prediction.py
# (make sure to edit 'path_model' and 'folder_path' to your own setup)


# project imports
import os
import numpy as np
from glob import glob
from ext.predict import predict

# User-defined paths
path_model = 'Put_your_path_here\trained_Models\Final_model_50_epochs.h5'
folder_path = 'Put_your_image_path_here'             # this is the path for the folder that contains the image/images you want to segment 
original_image_name=None                             # if the image has a specific name, the user can insert it here, otherwise the image ending with .nii will be used
target_res = 0.8                               
 
# Loop through subdirectories
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

for subfolder in subfolders:
    path_images = None

    if original_image_name:
        candidate_path = os.path.join(subfolder, original_image_name)
        if os.path.exists(candidate_path):
            path_images = candidate_path
        else:
            print(f"Specified original image '{original_image_name}' not found in {subfolder}. Skipping.")
            continue
    else:
        # Automatically find image not containing unwanted keywords
        nii_files = glob(os.path.join(subfolder, '*.nii.gz'))
        path_images = next(
            (f for f in nii_files 
            if all(x not in os.path.basename(f) for x in ['denoised', 'mask','segmented', 'posteriors','posterior','Wholebrain'])),
             None)

        if not path_images:
            print(f"No original image found in {subfolder}. Skipping.")
            continue

    print(f"Found image: {path_images}")

    # Extract base name without extensions
    base_name = os.path.basename(path_images).replace('.nii.gz', '').replace('.nii', '')

    # Generate output paths
    path_segm = os.path.join(subfolder, f"{base_name}_segmented.nii.gz")
    path_posteriors = os.path.join(subfolder, f"{base_name}_posteriors.nii.gz")
    path_vol = os.path.join(subfolder, f"{base_name}_volumes.csv")
    path_resampled = os.path.join(subfolder, f"{base_name}_resampled.nii.gz")
    
    print(f"Processing: {path_images}")
    output_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36])
	


	# We can now provide various parameters to control the preprocessing of the input.
	# First we can play with the size of the input. Remember that the size of input must be divisible by 2**n_levels, so the
	# input image will be automatically padded to the nearest shape divisible by 2**n_levels (this is just for processing,
	# the output will then be cropped to the original image size).
	# Alternatively, you can crop the input to a smaller shape for faster processing, or to make it fit on your GPU.
    cropping = 512
	# Finally, we finish preprocessing the input by resampling it to the resolution at which the network has been trained to
	# produce predictions. If the input image has a resolution outside the range [target_res-0.05, target_res+0.05], it will
	# automatically be resampled to target_res.


	# After the image has been processed by the network, there are again various options to postprocess it.
	# First, we can apply some test-time augmentation by flipping the input along the right-left axis and segmenting
	# the resulting image. In this case, and if the network has right/left specific labels, it is also very important to
	# provide the number of neutral labels. This must be the exact same as the one used during training.
    flip = True
    n_neutral_labels = 7
	# Second, we can smooth the probability maps produced by the network. This doesn't change much the results, but helps to
	# reduce high frequency noise in the obtained segmentations.
    sigma_smoothing = 0.0
	# Then we can operate some fancier version of biggest connected component, by regrouping structures within so-called
	# "topological classes". For each class we successively: 1) sum all the posteriors corresponding to the labels of this
	# class, 2) obtain a mask for this class by thresholding the summed posteriors by a low value (arbitrarily set to 0.1),
	# 3) keep the biggest connected component, and 4) individually apply the obtained mask to the posteriors of all the
	# labels for this class.
	# Example: (continuing the previous one)  generation_labels = [0, 24, 507, 2, 3, 4, 17, 25, 41, 42, 43, 53, 57]
	#                                             output_labels = [0,  0,  0,  2, 3, 4, 17,  2, 41, 42, 43, 53, 41]
	#                                       topological_classes = [0,  0,  0,  1, 1, 2,  3,  1,  4,  4,  5,  6,  7]
	# Here we regroup labels 2 and 3 in the same topological class, same for labels 41 and 42. The topological class of
	# unsegmented structures must be set to 0 (like for 24 and 507).
	# topology_classes = 'data/labels_classes_priors/synthseg_topological_classes.npy'
    topology_classes = None #output_labels

	# Finally, we can also operate a strict version of biggest connected component, to get rid of unwanted noisy label
	# patch that can sometimes occur in the background. If so, we do recommend to use the smoothing option described above.
    keep_biggest_component = False

	# Regarding the architecture of the network, we must provide the predict function with the same parameters as during
	# training.
    n_levels = 5
    nb_conv_per_level = 2
    conv_size = 3
    unet_feat_count = 24
    activation = 'elu'
    feat_multiplier = 2

    predict(path_images,
        path_segm,
	path_model,
	output_labels,
	n_neutral_labels=n_neutral_labels,
	path_posteriors=path_posteriors,
	path_resampled=path_resampled,
	path_volumes=path_vol,
	names_segmentation=None,
	cropping=cropping,
	target_res=target_res,
	flip=flip,
	topology_classes=topology_classes,
	sigma_smoothing=sigma_smoothing,
	keep_biggest_component=keep_biggest_component,
	n_levels=n_levels,
	nb_conv_per_level=nb_conv_per_level,
	conv_size=conv_size,
	unet_feat_count=unet_feat_count,
	feat_multiplier=feat_multiplier,
	activation=activation,
	gt_folder=None,
	posterior_class_to_save=[6],
	compute_distances=False)
	
