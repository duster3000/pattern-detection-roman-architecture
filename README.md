# Repeated pattern detection of architectural decoration in Roman Asia Minor

Code van de masterproef 'AI-gebaseerde patroonherkenning van architecturale decoratie in Romeins Klein Azië'

## Segmentation
semantic_discovery.ipynb and semantic_discovery_run.py are code from https://github.com/francesco-p/semantic-discovery with minor changes.

semantic_discovery_median_bilateral_filter_clahe.ipynb is the algorithm with preprocessing.

best_parameters_rgs.py is the script to determine the best parameters using random search.

segmentation.py is the algorithm that cuts out the segmentation masks. The output is saved in 'segmentation', each image has a separate folder.

The helper functions I made (with the help of ChatGPT) are in 'functions_semantic_discovery.py'. Other helper functions are in 'utils.py'.

## Frame Units using Bounding Boxes

main_boxes.ipynb and main_boxes.py is the algorithm that displays the boxes on a chosen image.

The helper functions are in 'functions_boxes.py'
adapted from https://github.com/kyusbok/Repeated-Pattern-Detection-using-CNN-activations/tree/master 
## Dataset
'trainset' is the set that has been labeled using LabelMe, the labels are in 'labels'.
'testset' are other random images from the dataset that have not been labeled.
The full dataset is available at: https://kuleuven-my.sharepoint.com/:f:/g/personal/thorsten_mahieu_student_kuleuven_be/EjY8uU-2cp1Pps8e5YtXlywBK-qvhozsq0GX1lPDvjUG8w?e=QnF808 