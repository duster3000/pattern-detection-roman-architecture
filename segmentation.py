import os
import sys 
sys.path.append('D:/semantic-discovery/src/')
import detector as de
import extractor as ex
import accumulator as ac
import utils
from functions import calculate_iou, compare_partition_with_labelme_annotation
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import networkx as nx

import itertools
import json
import random

# Define image and labelme annotation folder
image_folder = "D:/Thorsten M/Documenten/hoger onderwijs/4e jaar/masterproef/dataset/simpele bauornamentik afbeeldingen"
labelme_folder = "D:/Thorsten M/Documenten/hoger onderwijs/4e jaar/masterproef/dataset/labels"

# Define output folder
output_folder = 'd:/semantic-discovery/output/segmentation'

K = 5
RADIUS = 10
N_KEYPOINTS = 4000
SIGMA = 3
W = 9
TAU = 1.0
N_SUPERPIXELS = 210
ALPHA = 1.0
ksize = 3
clipLimit = 2.0
tileGridSize = (8,8)
D = 5
GAMMA = 0.7
COMPACTNESS = 22
keypoints_detection_method = 'canny'
params = {
        'K': K,
        'RADIUS': RADIUS,
        'TAU': TAU,
        'N_SUPERPIXELS': N_SUPERPIXELS,
        'N_KEYPOINTS': N_KEYPOINTS,
        'SIGMA': SIGMA,
        'W': W,
        'ALPHA': ALPHA,
        'ksize': ksize,
        'clipLimit': clipLimit,
        'tileGridSize': tileGridSize,
        'D': D,
        'GAMMA': GAMMA,
        'COMPACTNESS': COMPACTNESS,
        'keypoints_detection_method': keypoints_detection_method
    }

keypoints_detection_method = 'canny'


# Perform segmentation algorithm on each image using the parameters
for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)
    FILENAME = image_path
    original_img = cv2.imread(FILENAME)
    if not (os.path.isfile(image_file) or os.path.isfile(FILENAME) and original_img is not None):
        print(f"Error: Unable to read image file: {FILENAME}")
        continue
    else:
        original_img = cv2.imread(FILENAME, cv2.IMREAD_UNCHANGED)

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
        utils.spixel_segmentation_mask3(enhanced_img, best_partition_nodes, superpixels, output_folder, param_str, image_file )
        """
        pred, rmask = utils.spixel_segmentation_mask(img, best_partition_nodes, superpixels)
        rmask = (rmask * 255.).astype('uint8')
        overlayed_img = cv2.addWeighted(original_img, 0.99, overlay_bgr, 0.99, 0)
        mask_over_img = cv2.addWeighted(rmask, 0.4, overlayed_img, 0.4, 0)
        output_filename = os.path.join(output_folder, f"{param_str}_{os.path.basename(image_file)}")
        success = cv2.imwrite(os.path.join(output_folder, output_filename), mask_over_img)
        print(success)
        """

    else: 
        print("Error: Segmentation failed for image:", image_file)
        continue  
