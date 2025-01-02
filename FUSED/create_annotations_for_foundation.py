import json
import csv
import torch

# Load COCO annotations from JSON file
# with open('/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/Natural/cityscapes/cityscapes_source/annotations/image_info_test-dev2017.json', 'r') as f:
#     coco_data = json.load(f)
# annotation_path = '/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/BCD/c_view_data/common_cview_same_size/annotations/instances_val2017.json'
# annotation_path = '/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/GBC/GBCNet/annotations/image_info_test-dev2017.json'
with open(annotation_path, 'r') as f:
    coco_data = json.load(f)

# Initialize CSV writer and specify the output file path
output_path = '/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/GBC/GBCNet/test_id.csv'
with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write header row with column names
    writer.writerow(['image_id', 'image_names', 'class'])

    # Process each image in COCO dataset
    for img in coco_data['images']:
        # Target Vector (Has dimension equal to number of classes)
        class_num = coco_data['categories'].__len__()
        target_vector = torch.zeros(class_num)
        
        image_id = img['id']  # Get the image ID
        image_name = img['file_name']
        
        # Check if the image has annotations (bounding boxes)
        for ann in coco_data['annotations']:
            if ann['image_id'] == img['id']:
                if 'cview' in annotation_path:
                    ann['category_id'] = 0
                target_vector[ann['category_id']] = 1
        
        # Write
        target_vector = target_vector.tolist()
        if target_vector == [1]:
            target_vector.append(0)
        elif target_vector == [0]:
            target_vector.append(1)
        writer.writerow([image_id, image_name, target_vector])
