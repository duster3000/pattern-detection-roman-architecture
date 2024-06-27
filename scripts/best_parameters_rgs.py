"""
This script determines the best parameters for algorithm using random search and writes them to a CSV file.

"""

import os
import sys 
import csv
import random
import cv2
import numpy as np
from skimage.segmentation import slic
import networkx as nx

sys.path.append('./src/')
import detector as de
import extractor as ex
import accumulator as ac
import utils
from functions_semantic_discovery import *

# Define image and labelme annotation folder
image_folder = "./dataset/trainset"
labelme_folder = "./dataset/labels"

# Define output folder
output_folder = './output'

# Define parameter ranges
param_grid = {
    'keypoints_detection_method': ['canny', 'SIFT'],
    'K': [5, 7],
    'RADIUS': list(range(2, 153, 3)),
    'TAU': [1.0, 1.25, 1.5],
    'N_SUPERPIXELS': list(range(190, 295, 5)),
    'N_KEYPOINTS': list(range(4000, 10001, 1000)),
    'SIGMA': [2, 3, 4],
    'W': [9, 11, 13],
    'ALPHA': [1.0, 1.1],
    'ksize': [3, 5, 7, 9],
    'clipLimit': [2.0, 3.0, 4.0],
    'tileGridSize': [(8, 8), (16, 16)],
    'D': [3, 5, 7, 9, 11],
    'GAMMA': [0.6, 0.7, 0.8],
    'COMPACTNESS': list(range(20, 27)),
    'sigma_color': [50, 75],
    'sigma_space' : [50, 75]
    
}

best_score = -1  # Initialize best score
best_params = {}
jaccard_scores = []
best_params_dict = {}
num_samples = 150
average_iou_list = []

# Open a CSV file for logging
csv_file = open('/output/csv/parameter_log.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Image', 'Segment', 'Parameters', 'IoU'])

for index, _ in enumerate(range(num_samples)):
    random_params = {
        'K': random.choice(param_grid['K']),
        'RADIUS': random.choice(param_grid['RADIUS']),
        'TAU': random.choice(param_grid['TAU']),
        'N_SUPERPIXELS': random.choice(param_grid['N_SUPERPIXELS']),
        'N_KEYPOINTS': random.choice(param_grid['N_KEYPOINTS']),
        'SIGMA': random.choice(param_grid['SIGMA']),
        'W': random.choice(param_grid['W']),
        'ALPHA': random.choice(param_grid['ALPHA']),
        'ksize': random.choice(param_grid['ksize']),
        'clipLimit': random.choice(param_grid['clipLimit']),
        'tileGridSize': random.choice(param_grid['tileGridSize']),
        'D': random.choice(param_grid['D']),
        'GAMMA': random.choice(param_grid['GAMMA']),
        'COMPACTNESS': random.choice(param_grid['COMPACTNESS']),
        'keypoints_detection_method': random.choice(param_grid['keypoints_detection_method']),
        'sigma_color' : random.choice(param_grid['sigma_color']),
        'sigma_space' : random.choice(param_grid['sigma_space'])
    }
    # Set hyperparameters
    K = random_params['K'] 
    RADIUS = random_params['RADIUS'] 
    N_KEYPOINTS = random_params['N_KEYPOINTS'] 
    SIGMA = random_params['SIGMA'] 
    W = random_params['W'] 
    TAU = random_params['TAU'] 
    N_SUPERPIXELS = random_params['N_SUPERPIXELS'] 
    ALPHA = random_params['ALPHA'] 
    ksize = random_params['ksize'] 
    clipLimit = random_params['clipLimit'] 
    tileGridSize = random_params['tileGridSize'] 
    D = random_params['D']
    GAMMA = random_params['GAMMA']
    COMPACTNESS = random_params['COMPACTNESS']
    keypoints_detection_method = random_params['keypoints_detection_method']
    print(random_params)

    image_iou_list = []
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        FILENAME = image_path
        original_img = cv2.imread(FILENAME, cv2.IMREAD_UNCHANGED)
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

        sigma_color = random_params['sigma_color']
        sigma_space = random_params['sigma_space']
        enhanced_img = cv2.bilateralFilter(enhanced_img, D, sigma_color, sigma_space)

        enhanced_img = cv2.pow(enhanced_img, GAMMA)
        enhanced_img = img = (enhanced_img * 255).astype(np.uint8)

        kpdetector = de.KeypointsDetector()
        if keypoints_detection_method == 'canny':
            keypoints = kpdetector.canny(img, N_KEYPOINTS)
        else:
            sift = cv2.SIFT_create()
            keypoints = sift.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)

        extractor = ex.DescriptorExtractor()
        descriptors = extractor.daisy(keypoints, img)

        accumulator = ac.Accumulator(img)
        flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
        matches = flann.knnMatch(descriptors, descriptors, k=K)
        for i, m_list in enumerate(matches):
            o = (int(keypoints[i].pt[1]), int(keypoints[i].pt[0]))
            points = []
            rank = 1
            for m in m_list:
                d = (int(keypoints[m.trainIdx].pt[1]), int(keypoints[m.trainIdx].pt[0]))
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

        superpixels = slic(img, n_segments=N_SUPERPIXELS, compactness=COMPACTNESS, sigma=SIGMA)

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

        best_partition_nodes, best_partition = utils.findBestPartition_alt(adj_mat, alpha=ALPHA, viz=False)
        print('image shape: ', img.shape)
        if best_partition_nodes:
            print("Best partition nodes len:", len(best_partition_nodes))
            print("Best partition shape:", best_partition.shape)
            pred, rmask = generate_binary_masks(img, best_partition_nodes, superpixels)
            rmask = (rmask * 255.).astype('uint8')
            print("pred shape: ", pred.shape)
            print("rmask.shape: ", rmask.shape)
            mask_over_img = cv2.addWeighted(rmask, 0.5, img, 0.5, 0)
            print("mask_over_img.shape", mask_over_img.shape)
            score, segment_iou_list = compare_partition_with_labelme_annotation(labelme_folder, image_file, pred, index)
            print("IoU Score image:", score)
            image_iou_list.append(score)
            """
            param_str = '_'.join([f"{key}_{value}" for key, value in random_params.items()])
            output_filename = f"output_{param_str}.jpg"
            success = cv2.imwrite(os.path.join(output_folder, output_filename), mask_over_img)
            if not success:
                print(f"Failed to save output image: {output_filename}")
            else:
                print(f"Saved output image: {output_filename}")
            """    
            # Write IoU scores and parameters to CSV
            for segment_idx, segment_iou in enumerate(segment_iou_list):
                csv_writer.writerow([image_file, segment_idx, random_params, segment_iou])
            csv_writer.writerow([image_file, "all", random_params, score])

                
        else: 
            print("Error: Segmentation failed for image:", image_file)
            score = np.nan
            print("IoU Score image:", score)
            image_iou_list.append(score)
            continue

    average_iou_score = np.mean(image_iou_list)
    csv_writer.writerow(['ALL_IMAGES', '', random_params, average_iou_score])
    print("Average IoU Score across images:", average_iou_score)

    if average_iou_score > best_score:
        best_score = average_iou_score
        best_params = random_params
        param_str = '_'.join([f"{key}_{value}" for key, value in random_params.items()])
        print("new Best Parameters:", param_str)
        print("new Best Score:", best_score)
        csv_writer.writerow(['BEST', '', param_str, best_score])
        
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

# Close the CSV file
csv_file.close()

print("Best Parameters:", best_params)
print("Best Score:", best_score)
