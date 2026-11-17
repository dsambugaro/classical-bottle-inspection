"""Automated plastic bottle inspection with classical computer vision and classifiers.

Reproduces the inspections of the Kaggle "Water Bottle Defect-Level Detection"
dataset (originally solved with YOLOv8) using fixed regions of interest,
generic descriptors (HOG, LBP, HSV color) and classical classifiers. Each
descriptor is classified separately to compare their impact.

Modules, in pipeline order:

- config:        every constant (ROIs, descriptor settings, feature vectors)
- dataset:       loads the images and labels with the original Kaggle split
- preprocessing: background, lighting normalization, localization, crops
- features:      HOG, LBP and HSV color of each ROI and of the whole bottle
- extraction:    runs preprocessing + features over the dataset, with a cache
- models:        classifiers and their hyperparameter grids
- evaluation:    metrics, pass/reject criteria, alerts, localization error
- plots:         figures
- didactic:      optional figures of the real decision of each classifier
- pipeline:      orchestrates everything and writes the results
- cli:           command line interface
"""
