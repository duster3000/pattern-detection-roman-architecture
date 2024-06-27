import os
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms
from AlexNetConvLayers import alexnet_conv_layers
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageFilter
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.stats import multivariate_normal
from skimage.feature import peak_local_max
import cv2

#parameters
sigma_l = []
alfa_l = [5, 7, 15, 15, 15]  
fi_prctile = 80
delta = 0.65

subsample_pairs = 10
peaks_max = 10000
starting_index= 10

preprocess_transform = transforms.Compose([transforms.ToTensor()])

dev = torch.device("cuda")


def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    return preprocess_transform(image).unsqueeze(0).to(dev)

def compress_image_with_fixed_pixels(img_path, target_pixels):
    """
    Compress the image while maintaining aspect ratio and ensuring a fixed number of pixels.

    Args:
    - img_path (str): Path to the image file.
    - target_pixels (int): Desired number of pixels (width * height) of the compressed image.

    Returns:
    - str: Path to the compressed image file.
    """

    # Open the image
    with Image.open(img_path) as img:

        if img.mode == 'RGBA':
            img = img.convert('RGB')
        # Calculate the scale factor to resize the image
        original_size = img.size
        width, height = img.size
        #print("height ",height)
        current_pixels = width * height
        #print("current pixels ", current_pixels)
        scale_factor = (target_pixels / current_pixels) ** 0.5
        #print("scale factor ", scale_factor)
        # Calculate the new width based on the scale factor
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        #print("new height ", new_height)
        
        # Resize the image while maintaining the aspect ratio
        resized_img = img.resize((new_width, new_height), resample=Image.LANCZOS)
        resized_img = resized_img.convert('RGB')
        #compressed_image_path = "compressed_image.jpg"
        compressed_image = preprocess_transform(resized_img).unsqueeze(0).to(dev)
        # Save the resized image with compression
        
        return compressed_image, scale_factor, resized_img


def get_boxes(img_path):
    compressed_image, scale_factor, resized_img = compress_image_with_fixed_pixels(img_path, 260000)
    if compressed_image is not None:
        print("Image loaded")
    else:
        print("Error loading image")

    # # Load model


    model = alexnet_conv_layers()
    model.to(dev)
    print('Model loaded')

    # # Convolutional features computation

    #conv features computation

    print("Computing convolutional features...")
    print(compressed_image.shape)
    conv_feats = model(compressed_image)
    print("Convolutional features computed")

    # # Peaks extraction

    #peaks extraction
    print("Starting peaks extraction...")
    peaks = []
    for li, l in enumerate(conv_feats):
        print(f"Processing feature map {li + 1} out of {len(conv_feats)}")
        peaks.append([])
        maps = l.squeeze().detach().cpu().numpy()
        sigma_l.append((compressed_image.size(2) / maps.shape[1]) / 2)

        # #visualization
        for fi, fmap in enumerate(maps):
            fmap = np.array(Image.fromarray(fmap).resize((compressed_image.size(3), compressed_image.size(2))))
            #tmp_max = maximum_filter(fmap, 1)
            #max_coords = peak_local_max(tmp_max, 5)

            #plt.imshow(fmap)
            #plt.show()

            fmap = gaussian_filter(fmap, sigma=10)
            tmp_max = maximum_filter(fmap, 1)
            max_coords = peak_local_max(tmp_max, 5)

            #plt.imshow(fmap)
            #plt.show()
            peaks[li].append(max_coords[np.random.permutation(max_coords.shape[0])[:peaks_max]])
    
    #compute displacement set and voting space
    pickefile = "V_" + os.path.basename(img_path) + ".pkl"
    print("Computing displacement set and voting space...")
    if os.path.exists(pickefile):
        with open(pickefile, 'rb') as f:
            V = pickle.load(f)
    else:
        quant_r, quant_c = np.mgrid[0:compressed_image.size(2):1, 0:compressed_image.size(3):1]
        V = np.zeros(quant_r.shape)
        quant_rc = np.empty(quant_r.shape + (2,), dtype=np.float32)
        quant_rc[:, :, 0] = quant_r
        quant_rc[:, :, 1] = quant_c
        disps = []
        for li, p in enumerate(peaks):
            disps.append([])
            for fi, p2 in enumerate(p):
                # pairs_inds = np.asarray([(i, j) for i in range(p2.shape[0]) for j in range(p2.shape[0]) if i != j and j > i])
                pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])), dtype=np.uint8).T.reshape(-1, 2)
                pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
                if pairs_inds.shape[0] > 0:
                    tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
                    
                else:
                    tmp_disps = np.asarray([[]])
                if tmp_disps.size == 0:
                    continue
                tmp_disps = tmp_disps[np.random.permutation(tmp_disps.shape[0])[:subsample_pairs]]
                disps[li].append(tmp_disps)
                #tmp_disps è Dfl
                for ij, dij in enumerate(tmp_disps):
                    tmp_Vfiij = multivariate_normal.pdf(quant_rc, mean=dij
                                                        , cov=np.asarray([[sigma_l[li], 0]
                                                                        , [0, sigma_l[li]]], dtype=np.float32))
                    tmp_Vfiij /= tmp_disps.shape[0]
                    V += tmp_Vfiij

        with open(pickefile, 'wb') as handle:
            pickle.dump(V, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Displacement set and voting space computed")


    #find best step
    print("Finding best step...")
    starting_ind = starting_index
    # dstar = np.asarray(((V[:, 0] / np.arange(0, V.shape[0], 1))[starting_ind:].argmax() + starting_ind
    #                    , (V[0, :] / np.arange(0, V.shape[1], 1))[starting_ind:].argmax() + starting_ind))

    dstar = np.asarray((V[starting_ind:, 0].argmax() + starting_ind
                    , V[0, starting_ind:].argmax() + starting_ind))

    # Compute consistent votes to compute fi
    print("Computing consistent votes to compute fi...")
    fi_acc = []
    for li, p in enumerate(peaks):
        for fi, p2 in enumerate(p):
            pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])), dtype=np.uint8).T.reshape(
                -1, 2)
            pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
            if pairs_inds.shape[0] > 0:
                tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
            else:
                fi_acc.append(0)
                continue
            tmp_disps = tmp_disps[np.random.permutation(tmp_disps.shape[0])[:subsample_pairs]]
            fi_acc.append(len([1 for dij in tmp_disps if (np.linalg.norm(dij - dstar)) < 3 * alfa_l[li]]))
    print("Consistent votes computed")

    
    print("Computing percentile for fi...")
    param_fi = np.percentile(fi_acc, fi_prctile)
    print("Percentile computed")
    print(param_fi)


    #find weights for filters
    print("Finding weights for filters...")
    disps_star = []
    weights = []
    for li, p in enumerate(peaks):
        disps_star.append([])
        weights.append([])
        for fi, p2 in enumerate(p):
            # pairs_inds = np.asarray([(i, j) for i in range(p2.shape[0]) for j in range(p2.shape[0]) if i != j and j > i])
            pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])), dtype=np.uint8).T.reshape(
                -1, 2)
            pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
            if pairs_inds.shape[0] > 0:
                tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
            else:
                tmp_disps = np.asarray([[]])
            weights[li].append(0)
            if tmp_disps.size == 0:
                continue
            tmp_disps = tmp_disps[np.random.permutation(tmp_disps.shape[0])[:subsample_pairs]]
            disps_star[li].append(tmp_disps)
            # tmp_disps è Dfl

            for ij, dij in enumerate(tmp_disps):
                tmp_diff = np.linalg.norm(dij - dstar)
                if tmp_diff < 3 * alfa_l[li]:
                    # φ è 80esimo percentile, bisogna sommare i pesi per calcolare per ogni filtro
                    wijfl = np.exp(-(tmp_diff ** 2)
                                / (2 * (alfa_l[li] ** 2))) \
                            / (tmp_disps.shape[0] + param_fi)
                    weights[li][-1] += wijfl
    print("Weights computed")

    
    #find filters with weights higher than threshold
    print("Finding filters with weights higher than threshold...")
    selected_filters = []
    for li, w in enumerate(weights):
        tmp_weight_thr = delta * max(w)
        selected_filters.append([fi for fi, w2 in enumerate(w) if w2 > tmp_weight_thr])
    print("Filters found")


    #accumulate origin coordinates loss
    print("Accumulating origin coordinates loss...")
    acc_origin = []
    acc_origin_weights = []
    for li, w in enumerate(weights):
        for fi in selected_filters[li]:
            p2 = peaks[li][fi]
            pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])), dtype=np.uint8).T.reshape(
                -1, 2)
            pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
            if pairs_inds.shape[0] > 0:
                tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
            else:
                fi_acc.append(0)
                continue
            cons_disps = [dij for ij, dij in enumerate(tmp_disps)
                                        if (np.linalg.norm(dij - dstar)) < 3 * alfa_l[li]]
            cons_disps_weights = [np.exp(-(np.linalg.norm(dij - dstar) ** 2)/ (2 * (alfa_l[li] ** 2))) / (tmp_disps.shape[0] + param_fi)
                                for dij in cons_disps]
            acc_origin.extend(cons_disps)
            acc_origin_weights.extend(cons_disps_weights)
    print("Origin coordinates loss accumulated")

  
    # Find minimum loss
    print("Finding minimum loss...")
    o_r = np.linspace(-dstar[0], dstar[0], 10)
    o_c = np.linspace(-dstar[1], dstar[1], 10)
    min_rc = (-1, -1)
    min_val = np.inf
    for r in o_r:
        for c in o_c:
            tmp_orig = np.asarray([r, c])
            tmp_val = [np.linalg.norm(np.mod((dij - tmp_orig), dstar) - (dstar / 2)) * acc_origin_weights[ij]
                        for ij, dij in enumerate(acc_origin)]
            tmp_val = np.sum(tmp_val)
            if tmp_val < min_val:
                min_val = tmp_val
                min_rc = (r, c)
    print("Minimum loss found")

    
    # Generate boxes
    print("Generating boxes...")
    boxes = []
    tmp_img = np.array(resized_img)
    for ri in range(100):
        min_r = min_rc[0] + (dstar[0] * ri) - (dstar[1] / 2)
        if min_r > tmp_img.shape[0]:
            break
        for ci in range(100):
            min_c = min_rc[1] + (dstar[1] * ci) - dstar[0] / 2
            if min_c > tmp_img.shape[1]:
                break
            tmp_box = np.asarray([min_c, min_r, dstar[1], dstar[0]])
            boxes.append(tmp_box)
    print("Boxes generated")
    boxes = [box/scale_factor for box in boxes]  # Scale each box
    print(boxes)
    return boxes

def save_boxes_to_json(boxes, output_file_path):
    # Convert boxes to a JSON-serializable format
    boxes_serializable = [box.tolist() if isinstance(box, np.ndarray) else box for box in boxes]
    with open(output_file_path, 'w') as f:
        json.dump(boxes_serializable, f)
    print(f"Boxes saved to {output_file_path}")

def load_boxes_from_json(json_file_path):
    with open(json_file_path, 'r') as f:
        boxes_list = json.load(f)
    boxes = [np.array(box) for box in boxes_list]
    print(f"Boxes loaded from {json_file_path}")
    return boxes

def draw_bounding_boxes(ax, fig, coords, boxes, color):
    
    # Draw bounding boxes based on coords
    if coords is not None:
    
        min_x = int(coords['min_x'])
        min_y = int(coords['min_y'])
        max_x = int(coords['max_x'])
        max_y = int(coords['max_y'])
        width = max_x - min_x
        height = max_y - min_y
    
    # Draw vertical lines within the bounding box
        if boxes is not None and len(boxes) > 0: 
            # Draw a rectangle for the bounding box
            rect = Rectangle((min_x, min_y), width, height, linewidth=1.5, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            for bb in boxes:
                x = int(bb[0])
                box_width = int(bb[2])
                # Ensure the line is within the bounding box
                if min_x <= x <= max_x:
                    ax.plot([x, x], [min_y, max_y],  color=color, linewidth=1.5)
                    if min_x <= (x + box_width) <= max_x:
                        ax.plot([x + box_width, x + box_width], [min_y, max_y],  color=color, linewidth=1.5)

        plt.axis('off')

def template_match(original_image, template_image):
    result = cv2.matchTemplate(original_image, template_image, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc, max_val

# Function to generate a random color
def random_color():
    return (random.random(), random.random(), random.random())
