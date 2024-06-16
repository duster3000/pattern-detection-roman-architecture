# Repeated pattern detection of architectural decoration in Roman Asia Minor

This repository contains code and resources for the master's thesis on AI-based pattern recognition of architectural decoration in Roman Asia Minor.


## Segmentation
### Notebooks and Scripts

- 'semantic_discovery.ipynb' and 'semantic_discovery_run.py' are code from https://github.com/francesco-p/semantic-discovery with minor changes.

- 'semantic_discovery_median_bilateral_filter_clahe.ipynb' is the algorithm with preprocessing.

- 'best_parameters_rgs.py' is the script to determine the optimal parameters using random search.

- 'segmentation.py' is the algorithm that cuts out the segmentation masks. The output is saved in 'segmentation', each image has a separate folder.
### Helper Functions

The helper functions I made (with the help of ChatGPT) are in 'functions_semantic_discovery.py'. 
Other helper functions are in 'utils.py'.

## Frame Units using Bounding Boxes
### Notebooks and Scripts

- main_CNN_activations.py is the code from https://github.com/kyusbok/Repeated-Pattern-Detection-using-CNN-activations with added compression.

- 'main_boxes.ipynb' and 'main_boxes.py' contain the algorithm that displays the boxes on a chosen image.
### Helper Functions
- The helper functions are in 'functions_boxes.py'. It includes a version of main_CNN_activations in the form of a function and other functions I made with the help of ChatGPT.
## Combination
- 'combination.py' executes segmentation.py and main_boxes.py sequentially on one image using some of the optimal parameters.

## Dataset
- 'trainset' Contains labeled images using LabelMe. Labels are stored in the 'labels' folder
- 'testset' comprises other random images from the dataset that have not been labeled.
The full dataset is available at [this SharePoint link](https://kuleuven-my.sharepoint.com/:f:/g/personal/thorsten_mahieu_student_kuleuven_be/EjY8uU-2cp1Pps8e5YtXlywBK-qvhozsq0GX1lPDvjUG8w?e=QnF808).