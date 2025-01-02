import os
import numpy as np 
from torch.utils.data import DataLoader
from tqdm.auto import tqdm;
from PIL import Image
import copy
import random
from torchvision import transforms
import torch

class SingleImageDataset():
    def __init__(self, image_path, text_path, iou_threshold=0.1, topk=5, img_size=224):
        self.image_path = image_path
        self.text_path = text_path
        self.img_size = img_size

        self.proposals = self.load_proposals(text_path, iou_threshold, topk)

    def __len__(self):
        return 1  # Single image dataset

    def __getitem__(self, index):
        # Load image
        image = self.load_image(self.image_path)

        # Create crops from proposals
        crops = self.create_crops(image, self.proposals, self.img_size)

        return torch.stack(crops)

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

    def load_proposals(self, text_path, iou_threshold, topk):
        # Load proposals from text file
        boxes = np.loadtxt(text_path, dtype=np.float32)
        proposals = self.non_max_suppression(boxes, iou_threshold)
        if len(proposals) < topk:
            additional_proposals = random.choices(proposals, k=topk - len(proposals))
            proposals = np.concatenate([proposals, additional_proposals])
        proposals = proposals[:topk]
        return proposals

    def create_crops(self, image, proposals, img_size):
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        crop_list = []
        for box in proposals:
            pascal_box = self.convert_yolo_pascal(box[:4], image)
            pil_crop = image.crop(pascal_box)
            pil_crop = transform(pil_crop)
            crop_list.append(pil_crop)

        return crop_list

    def convert_yolo_pascal(self, box, image):
        W, H = image.size
        cx, cy, w, h = box
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        bbox = [x1, y1, x2, y2]
        return bbox

    def non_max_suppression(self, boxes, iou_threshold):
        boxes_copy = copy.deepcopy(boxes)
        while True:
            selected_indices = []
            removed_indices = []
            for i in range(len(boxes_copy)):
                if i in selected_indices or i in removed_indices:
                    continue
                current_box = boxes_copy[i]
                selected_indices.append(i)
                for j in range(i + 1, len(boxes_copy)):
                    if j in selected_indices or j in removed_indices:
                        continue
                    if self.calculate_iou(current_box, boxes_copy[j]) > iou_threshold:
                        removed_indices.append(j)
            
            selected_indices = sorted(selected_indices)
            if len(selected_indices) == len(boxes_copy):
                break
            else:
                boxes_copy = boxes_copy[selected_indices]
        return boxes_copy

    def calculate_iou(self, box1, box2):
        x1, y1, w1, h1, c1 = box1
        x2, y2, w2, h2, c2 = box2
        
        # Convert to absolute coordinates
        x1 = x1 - w1 / 2
        y1 = y1 - h1 / 2
        x2 = x2 - w2 / 2
        y2 = y2 - h2 / 2
        
        # Calculate intersection coordinates
        x_intersection = max(x1, x2)
        y_intersection = max(y1, y2)
        w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
        h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)
        
        # Calculate intersection area
        intersection_area = w_intersection * h_intersection
        
        # Calculate areas of the bounding boxes
        area1 = w1 * h1
        area2 = w2 * h2
        
        iou = intersection_area / float(area1 + area2 - intersection_area)
        return iou

if __name__ == '__main__':
    text_path = "/home/tajamul/scratch/DA/Datasets/Foundation_Data/INBreast/test2017_focalnet_a25/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm_preds.txt"
    image_path = "/home/tajamul/scratch/DA/Datasets/Foundation_Data/INBreast/test2017/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm.png"
    
    dataset = SingleImageDataset(image_path, text_path, topk=5)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in data_loader:
        print(batch.shape)  # This will print the shape of the batch containing image crops
