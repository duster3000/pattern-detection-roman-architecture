import os
import cv2
import numpy as np
import json
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compare_partition_with_labelme_annotation(labelme_folder, image_file, partition):
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
    output_folder= "../output/comparison"
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
            output_filename = os.path.join(output_folder, f"comparison_{label}_{os.path.basename(image_file)}")
            cv2.imwrite(output_filename, comparison_image)
            print(f"Saved comparison image: {output_filename}")

        iou = img_iou = np.mean(segment_iou_list)
    else:
        print(f"Annotation not found for image: {image_file}")
        iou = 0.0

    return iou, segment_iou_list

