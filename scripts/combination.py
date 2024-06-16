"""
this script detects patterns and saves the segmentation masks as separate images
"""

import os
import sys 
sys.path.append('./src/')
import detector as de
import extractor as ex
import accumulator as ac
import utils
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import networkx as nx
from functions_semantic_discovery import *
import json
import csv
import functions_boxes
# Define image and labelme annotation folder
image_folder = "./dataset/trainset"
labelme_folder = "./dataset/labels"

# Define output folder
output_folder = './segmentation'

image_file = "DSC_1054 2023-10-02 21_37_15.JPG"
# Perform segmentation algorithm on each image using the parameters
image_path = os.path.join(image_folder, image_file)
FILENAME = image_path
original_img = cv2.imread(FILENAME)
if not (os.path.isfile(image_file) or os.path.isfile(FILENAME) and original_img is not None):
    print(f"Error: Unable to read image file: {FILENAME}")
else:
    original_img = cv2.imread(FILENAME, cv2.IMREAD_UNCHANGED)

with open('best_parameter_log.csv', mode='r') as file:
    reader = csv.DictReader(file)

    # Read each row and append the parameters to the list
    for row in reader:
        # Convert relevant values from strings to appropriate types
        params = {
            'K': int(row['K']),
            'RADIUS': int(row['RADIUS']),
            'TAU': float(row['TAU']),
            'N_SUPERPIXELS': int(row['N_SUPERPIXELS']),
            'N_KEYPOINTS': int(row['N_KEYPOINTS']),
            'SIGMA': int(row['SIGMA']),
            'W': int(row['W']),
            'ALPHA': float(row['ALPHA']),
            'ksize': int(row['ksize']),
            'clipLimit': float(row['clipLimit']),
            'tileGridSize': (int(row['tileGridSize_D']), int(row['tileGridSize_G'])),
            'D': int(row['D']),
            'GAMMA': float(row['GAMMA']),
            'COMPACTNESS': int(row['COMPACTNESS']),
            'keypoints_detection_method': row['keypoints_detection_method'],
            'IoU_Score': float(row['IoU_Score'])
        }

        K = params['K']
        RADIUS = params['RADIUS']
        N_KEYPOINTS = params['N_KEYPOINTS']
        SIGMA = params['SIGMA']
        W = params['W']
        TAU = params['TAU']
        N_SUPERPIXELS = params['N_SUPERPIXELS']
        ALPHA = params['ALPHA']
        ksize = params['ksize']
        clipLimit = params['clipLimit']
        tileGridSize = params['tileGridSize']
        D = params['D']
        GAMMA = params['GAMMA']
        COMPACTNESS = params['COMPACTNESS']
        keypoints_detection_method = params['keypoints_detection_method']


        # Apply Filtering and CLAHE
        median_img = cv2.medianBlur(original_img, ksize)

        lab_img = cv2.cvtColor(median_img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        lab_img[:, :, 0] = clahe.apply(lab_img[:, :, 0])
        enhanced_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
        enhanced_img = enhanced_img.astype(np.float32) / 255.0

        sigma_color = 75  # Filter sigma in the color space
        sigma_space = 75  # Filter sigma in the coordinate space
        enhanced_img = cv2.bilateralFilter(enhanced_img, D, sigma_color, sigma_space)

        # Apply gamma correction to further adjust contrast
        enhanced_img = cv2.pow(enhanced_img, GAMMA)
        enhanced_img = img = (enhanced_img * 255).astype(np.uint8)

        # Perform keypoint detection
        kpdetector = de.KeypointsDetector()
        if keypoints_detection_method == 'canny':
            keypoints = kpdetector.canny(img, N_KEYPOINTS)
        else:
            sift = cv2.SIFT_create()
            keypoints = sift.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        # Extract descriptors
        extractor = ex.DescriptorExtractor()
        descriptors = extractor.daisy(keypoints, img)

        # Accumulate votes
        accumulator = ac.Accumulator(img)
        flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
        matches = flann.knnMatch(descriptors, descriptors, k=K)
        for i, m_list in enumerate(matches):
            o = (int(keypoints[i].pt[1]), int(keypoints[i].pt[0]))
            points = []
            rank = 1
            for m in m_list:
                d = (int(keypoints[m.trainIdx].pt[1]), int(keypoints[m.trainIdx].pt[0]))
                # Remove points near the source
                if utils.eu_dist(o, d) > RADIUS:
                    points.append(d)
                    accumulator.add_vote(o, d, rank, ksize=W)
                    rank += 1
            accumulator.add_splash(o, points)

        x, y = np.where(accumulator.accumulator > TAU)
        vote_list = accumulator.votes.copy()
        idx1 = np.nonzero(np.isin(vote_list[:, 0], x))[0]
        idx2 = np.nonzero(np.isin(vote_list[idx1, 1], y))[0]
        idx3 = idx1[idx2]
        mask = np.zeros(vote_list.shape[0])
        coords = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        for coord in coords:
            idx4 = np.where(np.all(vote_list[idx3, 0:2] == coord, axis=1))[0]
            mask[idx3[idx4]] = 1
        vote_list = vote_list[np.nonzero(mask)[0]]

        # Perform superpixel segmentation
        superpixels = slic(img, n_segments=N_SUPERPIXELS, compactness=COMPACTNESS, sigma=SIGMA)
        overlay = mark_boundaries(img, superpixels, color=(16/255, 124/255, 255/255), mode="thick")
        overlay_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Build adjacency matrix
        connections = []
        for vote in vote_list:
            pi1 = vote[2:4]
            pi2 = vote[4:6]
            connections.append((superpixels[pi1[0], pi1[1]], superpixels[pi2[0], pi2[1]]))
        n_clusters = superpixels.max() + 1

        adj_mat = np.zeros((n_clusters, n_clusters))
        for c in connections:
            adj_mat[c[0], c[1]] += 1
            adj_mat[c[1], c[0]] += 1
        np.fill_diagonal(adj_mat, 0)

        G = nx.from_numpy_array(adj_mat)
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        weights = np.array(weights)
        weights /= weights.max()
        maxw = weights.max()
        weights *= 10

        # Find best partition
        best_partition_nodes, best_partition = utils.findBestPartition_alt(adj_mat, alpha=ALPHA, viz=False)
        if best_partition_nodes:
            # Perform segmentation and save
            param_str = '_'.join([f"{key}_{value}" for key, value in params.items()])
            original_image_basename = os.path.splitext(os.path.basename(image_path))[0]
            segment_folder = os.path.join(output_folder,original_image_basename)
            if not os.path.exists(segment_folder):
                os.makedirs(segment_folder)
            create_and_save_segmented_masks(enhanced_img, best_partition_nodes, superpixels, segment_folder, param_str, image_file )


        else: 
            print("Error: Segmentation failed for image:", image_file)

# Load original image
original_image_path = os.path.join(image_folder, image_file)
original_image = original_img.copy()
original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# List of paths to segmented images
original_image_basename = os.path.splitext(os.path.basename(original_image_path))[0]
segment_folder = f"./segmentation/{original_image_basename}/"
segment_images_paths = [os.path.join(segment_folder, fname) for fname in os.listdir(segment_folder) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Convert the original image to RGB for displaying with matplotlib
image_np = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(1)
ax.imshow(image_np)

# Process each segment image
for segment_image_path in segment_images_paths:
    segment_base = os.path.splitext(segment_image_path)[0]  # Get the base name of the segment_image_path
    lines_boxes_image_path = segment_base + "_lines_boxes.png"
    lines_boxes_path = segment_base + "_lines_boxes.json"
    coords_json_path = segment_base + "_coords.json"  
    
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
