import os
import cv2
import numpy as np
import seaborn as sns
import json
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compare_partition_with_labelme_annotation(labelme_folder, image_file, partition, index ):
    """
    Compare a partition with the LabelMe annotation for a given image.

    Args:
    - labelme_folder (str): Path to the folder containing image annotation files.
    - image_file (str): Filename of the image.

    Returns:
    - float: Intersection over Union (IoU) between the partition and the LabelMe annotation.
    """
    annotation_file = os.path.splitext(image_file)[0] + '.json'
    annotation_path = os.path.join(labelme_folder, annotation_file)
    output_folder= "./output/comparison"
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f: # Load labelme annotation
            data = json.load(f)

        # Create a dictionary to store shapes for each label
        label_shapes_dict = {}
        for shape in data['shapes']:
            label = shape['label']
            if label not in label_shapes_dict:
                label_shapes_dict[label] = []
            label_shapes_dict[label].append(shape)

        segment_iou_list = []
        for label, shapes in label_shapes_dict.items():
            # Create a mask containing all shapes with the same label
            label_mask = np.zeros(partition.shape[1:], dtype=np.uint8)
            for s in shapes:
                pts = np.array(s['points'], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(label_mask, [pts], 255)

            best_segment_iou = 0
            best_segment = np.zeros(partition.shape[1:], dtype=np.uint8)
            for segment in partition:
                iou = calculate_iou(label_mask, segment)
                if iou > best_segment_iou:
                    best_segment_iou = iou
                    best_segment = segment
            segment_iou_list.append(best_segment_iou)

            # Save comparison image
            comparison_image = np.hstack((best_segment, label_mask))
            output_filename = os.path.join(output_folder, f"comparison_{label}_{index}_{os.path.basename(image_file)}")
            success = cv2.imwrite(output_filename, comparison_image)
            if success:
                print(f"Saved comparison image: {output_filename}")

        iou = img_iou = np.mean(segment_iou_list)
    else:
        print(f"Annotation not found for image: {image_file}")
        iou = 0.0

    return iou, segment_iou_list


def generate_binary_masks(img, superpixel_clusters, segments):
    """Creates binary mask for each pattern and combines them into a single mask and an array of masks.
    
    Args:
        img (numpy.ndarray): Input image.
        superpixel_clusters (list of list of int): List of clusters where each cluster is a list of superpixel indices.
        segments (numpy.ndarray): Array where each pixel value corresponds to a superpixel label.
    
    Returns:
        tuple: 
            - numpy.ndarray: An array of binary masks, one for each cluster.
            - numpy.ndarray: An image where each superpixel is colored according to its cluster.
    """
    
    # Initialize color palette for visualization
    sns.set_palette(sns.color_palette("Set1", n_colors=len(superpixel_clusters)))
    color_palette = sns.color_palette()
    
    # Prepare masks
    img_height, img_width = img.shape[:-1]
    colored_mask = np.zeros(img.shape, dtype=np.uint8)
    binary_masks = np.zeros((len(superpixel_clusters), img_height, img_width), dtype=np.uint8)
    
    # Create masks for each cluster
    for i, superpixels in enumerate(superpixel_clusters):
        temp_mask = np.zeros(img.shape[:-1], dtype=np.uint8)
        for spixel in superpixels:
            colored_mask[np.where(segments == spixel)] = np.array(color_palette[i]) * 255
            temp_mask[np.where(segments == spixel)] = 255
        binary_masks[i] = temp_mask

    return binary_masks, colored_mask


def create_and_save_segmented_masks(img, superpixel_clusters, segments, output_folder, param_str, image_file):
    """Creates segmentation mask for each cluster and saves it as a separate image and coordinate file.
    
    Args:
        img (numpy.ndarray): Input image.
        superpixel_clusters (list of list of int): List of clusters where each cluster is a list of superpixel indices.
        segments (numpy.ndarray): Array where each pixel value corresponds to a superpixel label.
        output_folder (str): Directory to save the output files.
        param_str (str): String identifier for parameter settings.
        image_file (str): Filename of the input image for reference.
    """

    sns.set_palette(sns.color_palette("Set1", n_colors=len(superpixel_clusters)))

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for i, superpixels in enumerate(superpixel_clusters):
        mask = np.zeros_like(segments, dtype=np.uint8)
        for spixel in superpixels:
            mask[segments == spixel] = 255

        # Find non-zero mask coordinates
        y_coords, x_coords = np.where(mask == 255)
        if len(x_coords) == 0 or len(y_coords) == 0:
            print(f"No non-zero mask for segment {i}, skipping.")
            continue

        # Calculate bounding box coordinates
        min_x = int(np.percentile(x_coords, 5))
        max_x = int(np.percentile(x_coords, 95))
        min_y = int(np.percentile(y_coords, 5))
        max_y = int(np.percentile(y_coords, 95))

        # Create bounding box mask
        bbox_mask = np.zeros_like(mask, dtype=np.uint8)
        bbox_mask[min_y:max_y+1, min_x:max_x+1] = 255

        # Apply bounding box mask to the original image
        masked_img = cv2.bitwise_and(img, img, mask=bbox_mask)

        # Crop the masked image to the bounding box
        cropped_masked_img = masked_img[min_y:max_y+1, min_x:max_x+1]

        # Save the coordinates in a file
        coord_file = os.path.join(output_folder, f"segment_{i}_{param_str}_coords.json")
        with open(coord_file, "w") as f:
            json.dump({"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}, f)

        # Save the cropped masked image as a separate file
        output_filename = os.path.join(output_folder, f"segment_{i}_{param_str}.jpg")
        cv2.imwrite(output_filename, cropped_masked_img)

        print(f"Saved image: {output_filename} with coordinates saved to {coord_file}")