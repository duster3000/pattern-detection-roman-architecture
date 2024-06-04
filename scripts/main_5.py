import cv2
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random
import functions_boxes


# Load original image
original_image_path = "../dataset/trainset/Parion3.png"
original_image = cv2.imread(original_image_path)
original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
segment_images_path = "../segmentation"
# List of paths to segmented images
segment_images_paths = [
    "D:/semantic-discovery/output/segmentation/segment_0_K_7_RADIUS_50_TAU_1.25_N_SUPERPIXELS_210_N_KEYPOINTS_6000_SIGMA_3_W_9_ALPHA_1.0_ksize_3_clipLimit_2.0_tileGridSize_(8, 8)_D_7_GAMMA_0.7_COMPACTNESS_22_keypoints_detection_method_canny_Parion3.png",
    "D:/semantic-discovery/output/segmentation/segment_2_K_5_RADIUS_10_TAU_1.0_N_SUPERPIXELS_210_N_KEYPOINTS_4000_SIGMA_3_W_9_ALPHA_1.0_ksize_3_clipLimit_2.0_tileGridSize_(8, 8)_D_5_GAMMA_0.7_COMPACTNESS_22_keypoints_detection_method_canny_Parion3.png",
    "D:/semantic-discovery/output/segmentation/segment_1_K_7_RADIUS_50_TAU_1.25_N_SUPERPIXELS_210_N_KEYPOINTS_6000_SIGMA_3_W_9_ALPHA_1.0_ksize_3_clipLimit_2.0_tileGridSize_(8, 8)_D_7_GAMMA_0.7_COMPACTNESS_22_keypoints_detection_method_canny_Parion3.png",
    "D:/semantic-discovery/output/segmentation/segment_3_K_7_RADIUS_20_TAU_1.5_N_SUPERPIXELS_210_N_KEYPOINTS_4000_SIGMA_4_W_13_ALPHA_1.0_ksize_3_clipLimit_2.0_tileGridSize_(8, 8)_D_5_GAMMA_0.7_COMPACTNESS_22_keypoints_detection_method_canny_Parion3.png",
    "D:/semantic-discovery/output/segmentation/segment_3_K_5_RADIUS_10_TAU_1.0_N_SUPERPIXELS_210_N_KEYPOINTS_4000_SIGMA_3_W_9_ALPHA_1.0_ksize_3_clipLimit_2.0_tileGridSize_(8, 8)_D_5_GAMMA_0.7_COMPACTNESS_22_keypoints_detection_method_canny_Parion3.png"

]
image_np = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(1)
ax.imshow(image_np)
# Process each segment image
for segment_image_path in segment_images_paths:
    segment_base = os.path.splitext(segment_image_path)[0]  # Get the base name of the segment_image_path
    lines_boxes_image_path = segment_base + "_lines_boxes.png"
    lines_boxes_path = segment_base + "_lines_boxes.json"
    coords_json_path = segment_image_path + ".json"
    
    # Check if the segment has a _lines_boxes image
    if not os.path.exists(lines_boxes_path):
        print(f"{lines_boxes_path} not found. Using CNN activations algorithm to create boxes.")
        boxes = functions_boxes.get_boxes(segment_image_path)
        # Save the _lines_boxes to a text file
        functions_boxes.save_boxes_to_json(boxes, lines_boxes_path)
       
    else:
        print(f"{lines_boxes_image_path} found. Using _lines_boxes image.")
        # Load the _lines_boxes 
        boxes = functions_boxes.load_boxes_from_json(lines_boxes_path)
    # Check if the segment has a coordinate JSON file
    if os.path.exists(coords_json_path):
        with open(coords_json_path, "r") as f:
            coords = json.load(f)
            min_x = coords["min_x"]
            max_x = coords["max_x"]
            min_y = coords["min_y"]
            max_y = coords["max_y"]
    else:
        print(f"{coords_json_path} not found. Using template matching to get coordinates.")
        template_image = cv2.imread(segment_image_path)
        template_image_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        top_left, match_val = functions_boxes.template_match(original_image_gray, template_image_gray)
        print(f"Match value: {match_val}")
        
        # Define bounding box
        h, w = template_image_gray.shape
        min_x, max_x = top_left[0], top_left[0] + w
        min_y, max_y = top_left[1], top_left[1] + h
        coords = {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}
        # Save coordinates to a JSON file
        with open(coords_json_path, "w") as f:
            json.dump(coords, f)

    # Draw bounding box on the original image
    color = functions_boxes.random_color()
    functions_boxes.draw_bounding_boxes(ax, fig, coords, boxes, color)
    
plt.show()
