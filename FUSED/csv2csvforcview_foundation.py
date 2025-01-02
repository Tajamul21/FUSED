import csv

# Read the original CSV file
# input_file = '/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/BCD/c_view_data/irch_gt.csv'
# output_file = '/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/BCD/c_view_data/common_cview_same_size/val_id.csv'

input_file = '/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/GBC/GBC_centre/annotations/train.csv'
output_file = '/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/GBC/GBC_centre/train_id.csv'

# input_file = '/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/GBC/GBC_centre/annotations/instances_val2017.json'
# output_file = '/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/GBC/GBC_centre/val_id.csv'

# input_file = '/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/GBC/GBC_centre/annotations/image_info_test-dev2017.json'
# output_file = '/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/GBC/GBC_centre/test_id.csv'

# Open the input and output files
with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
    # import ipdb; ipdb.set_trace()
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    
    # Write the header for the output file
    writer.writerow(['image_id', 'image_names', 'class'])
    
    # Process each row from the input file and write to the output file
    for image_id, row in enumerate(reader):
        img_path = row['img_path']
        label = int(row['label'])

        img_path = img_path.split('/')[-1]
        
        # Convert label to the required class format
        if label == 0:
            class_str = '[1.0, 0]'
        else:
            class_str = '[0, 1.0]'
        
        # Write the row to the output file
        writer.writerow([image_id, img_path, class_str])

print("CSV file created successfully.")
