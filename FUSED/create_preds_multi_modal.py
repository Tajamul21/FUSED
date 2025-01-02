import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset

from util.visualizer import COCOVisualizer
from util import box_ops
from tqdm import tqdm
from util.visualizer import COCOVisualizer
import cv2
import torchvision.transforms as transforms
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import datasets.transforms as T
from PIL import Image
from datasets.coco import make_coco_transforms
import glob
from tqdm import tqdm

           

def annotated_image(image_path, bboxes):
    # import pdb; pdb.set_trace()
    image = cv2.imread(image_path, -1)
    H, W  = image.shape[:2]
    for i,box in enumerate(bboxes):
        cx, cy, w, h = box
        x1 = int((cx-w/2)*W); x2 = int((cx+w/2)*W)
        y1 = int((cy-h/2)*H); y2 = int((cy+h/2)*H)
        box = [x1, y1, x2, y2]
        image = cv2.rectangle(image, box[:2], box[2:], (255, 0, 0), 5)
    return image

def save_preds(data_path, args, split, file_ext):
    # image_paths = glob.glob(os.path.join(data_path)+"/*/*.png")
    # image_paths = glob.glob(os.path.join(data_path)+"/*.png")
    # image_paths = glob.glob(os.path.join(data_path)+"/*.jpg")
    image_paths = glob.glob(os.path.join(data_path)+file_ext)
    # image_paths = glob.glob(os.path.join(data_path,split)+"/*/*.png")
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # postprocessors['bbox'].nms_iou_threshold = 1.0
    for i, image_path in enumerate(tqdm(image_paths)):
        image_array = Image.open(image_path).convert("RGB") 
        image, _ = transform(image_array, None)

        output = model.cuda()(image[None].cuda())
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
        output['scores'] = output['scores'].cpu().unsqueeze(1)
        output['labels'] = output['labels'].cpu().unsqueeze(1)
        output['boxes'] = box_ops.box_xyxy_to_cxcywh(output['boxes']).cpu()
        pred_data = torch.cat((output['boxes'], output['scores'], output['labels']), 1).numpy()
        
        # pred_data_path = image_path.replace(f"/{split}/", f"/{split}_focalnet/")[:-4]+"_preds.txt"
        pred_data_path = os.path.join(f"{data_path}"+"_focalnet_crops/",image_path.split("/")[-1][:-4]+"_preds.txt")
        os.makedirs("/".join(pred_data_path.split("/")[:-1]), exist_ok=True)
        # print(image_path, pred_data_path)
        np.savetxt(pred_data_path, pred_data)


def create_pred_data(data_path):
    pred_list = []
    breast_dirs = os.listdir(data_path)
    for i,folder in enumerate(tqdm(breast_dirs)):
        folder_path = os.path.join(data_path, folder)
        images = glob.glob(folder_path+"/*.png")
        for j,image_path in enumerate(images):
            item_info = {}
            preds_path = image_path[:-4]+"_preds.txt"
            preds = torch.tensor(np.loadtxt(preds_path))
            if(len(preds.shape)==1): preds = preds.unsqueeze(0)
            if(preds.shape[1]!=0):
                # print(preds.shape)
                # import pdb; pdb.set_trace()
                output = {"boxes": preds[:,:4], 
                        "scores": preds[:,4],
                        "labels": torch.zeros(preds.shape[0])}
            else:
                output = {"boxes": torch.tensor([[0,0,0,0]]), 
                        "scores": torch.tensor([0]),
                        "labels": torch.zeros(0)}
                # import pdb; pdb.set_trace()
            target_path = image_path[:-4]+".txt"
            if(os.path.isfile(target_path)):
                targets = torch.tensor(np.loadtxt(target_path))
                if(targets.shape[0]!=0):
                    if (len(targets.shape)==1): targets = targets.unsqueeze(0)
                    targets=targets[:,1:] 
                else:
                    print(folder)
                    targets=torch.tensor([])
            else:
                targets = torch.tensor([])
            item_info['pred'] = output
            item_info['target'] = {"boxes":targets}
            pred_list.append(item_info)
    return pred_list


# from calc_metrics2 import calc_froc, calc_accuracy
from medical_evaluation import calc_froc, calc_accuracy

def evaluate(data_path):
    pred_list = create_pred_data(data_path)
    # import pdb; pdb.set_trace()
    print(len(pred_list))
    fpi, sens = calc_froc(pred_list)
    tpr, fpr = calc_accuracy(pred_list)
    return tpr, fpr, fpi, sens


def save_plot_values(model_name, dataset, data):
    tpr, fpr, fpi, sens = data
    # import pdb; pdb.set_trace()
    auc_target_path = os.path.join("AUC","{}_{}_auc".format(dataset, model_name))
    np.save(auc_target_path, np.array([fpr,tpr]))
    
    froc_target_path = os.path.join("FROC","{}_{}_froc".format(dataset, model_name))
    np.save(froc_target_path, np.array([fpi,sens]))
    



if __name__ == '__main__':
    # # ROOT = "/home/kshitiz/scratch/FocalNet-DINO/exps/inbreast_fold_5/"
    # # ROOT = "/home/kshitiz/scratch/FocalNet-DINO/exps/inbreast_ddsm_4k_new/"
    # # ROOT = "/home/kshitiz/scratch/FocalNet-DINO/exps/ddsm_aiims_2k_new/"
    # ROOT = "/home/kshitiz/scratch/FocalNet-DINO/exps/ddsm_cbis_coco/"
    # ROOT = "/home/kshitiz/scratch/FocalNet-DINO/exps/ddsm_cbis_coco/"
    # # ROOT = "/home/kshitiz/scratch/FocalNet-DINO/exps/pre/"
    # ROOT = "/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_CHECKPOINTS/source_only/Focalnet/aiims"

    ################ VARIABLES TO SET ##################
    ####### DATA PATH FOR WHICH U WANT THE CROPS #######

    # DATA_PATH = "/home/keerti_km/scratch/Negroni_datasets/Coco_Data/BCD/AIIMS/train2017"
    # DATA_PATH = "/home/keerti_km/scratch/Negroni_datasets/Coco_Data/BCD/AIIMS/val2017"
    # model_config_path = "/home/keerti_km/scratch/another-sfda-reproduced/SFDA-NeurIPS-24/FUSED/weights/source_aiims/config_cfg.py"
    # model_checkpoint_path = "/home/keerti_km/scratch/another-sfda-reproduced/SFDA-NeurIPS-24/FUSED/weights/source_aiims/source_in.pth"
    
    DATA_PATH = "/home/keerti_km/scratch/Negroni_datasets/Coco_Data/BCD/INBreast/test2017"
    model_config_path = "/home/keerti_km/scratch/another-sfda-reproduced/SFDA-NeurIPS-24/FUSED/weights/source_aiims/config_cfg.py"
    model_checkpoint_path = "/home/keerti_km/scratch/another-sfda-reproduced/SFDA-NeurIPS-24/FUSED/weights/source_aiims/source_in.pth"

    ####################################################
    
    # model_config_path = os.path.join(ROOT,"Target_INBreast/config_cfg.py") # change the path of the model config file
    # model_checkpoint_path = os.path.join(ROOT,"source_in.pth")
    
    args = SLConfig.fromfile(model_config_path)

    args.lambda_eckd = 1.0
    args.temp = 1.0
    args.memory_bank_size = 1

    # See your dataset arrangement for this
    file_ext = "/*.png" #/*/*.png
    ####################################################

    args.device = 'cuda' 
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()
    args.fix_size = True 

    # DATA_PATH = "/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/Natural/sim10k/val2017"
    # DATA_PATH = "/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/Natural/kitti/val2017"
    # DATA_PATH = "/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/BCD/c_view_data/common_cview_same_size/val2017"
    # DATA_PATH = "/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/GBC/GBCNet/test2017"
    # TARGET_DATA_PATH = "/home/keerti_km/scratch/Negroni_datasets/Coco_Data/Natural/kitti/train2017"
    # DATA_PATH = "/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/GBC/GBC_centre/val2017"
    # DATA_PATH = "/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/GBC/GBC_centre/test2017"
    # import ipdb; ipdb.set_trace()
    save_preds(DATA_PATH, args, "Train", file_ext=file_ext)
    # save_preds(DATA_PATH, args, "Test")
    # save_preds2(TEST_path, args)    
    # evaluate(TRAIN_path)
    
