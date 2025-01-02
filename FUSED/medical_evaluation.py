import argparse
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
import matplotlib.pyplot as plt
from util.visualizer import COCOVisualizer

from main import get_args_parser


def get_preds(dataset):
    preds = []
    postprocessors['bbox'].nms_iou_threshold = 0.25
    for i, x in enumerate( tqdm(dataset) ):
        item_info = {}
        image, targets = dataset[i]
        output = model.cuda()(image[None].cuda())
        model_output = output
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
        output['scores'] = output['scores'].cpu()
        output['labels'] = output['labels'].cpu()
        output['boxes'] = box_ops.box_xyxy_to_cxcywh(output['boxes']).cpu()
        item_info['image'] = image
        item_info['target'] = targets
        item_info['pred'] = output
        preds.append(item_info)
    
    return preds, model_output # These preds have the boxes in cx,cy,w,h format ultimately


# from netcal.metrics import ECE

def does_it_match_with_gt(gt, pred):
    # If center of pred is inside the gt, it is a true positive
    box_pascal_gt = ( gt[0]-(gt[2]/2.) , gt[1]-(gt[3]/2.), gt[0]+(gt[2]/2.), gt[1]+(gt[3]/2.) )
    if (pred[0] >= box_pascal_gt[0] and pred[0] <= box_pascal_gt[2] and
            pred[1] >= box_pascal_gt[1] and pred[1] <= box_pascal_gt[3]):
        return True
    return False


def get_D_ECE_and_auc(pred_data, threshold=0.1, n_bins=[10, 10, 10, 10, 10]):
    # ece = ECE(n_bins, detection=True)
    scores_for_all_images = []
    is_confident_box_matched = []
    ground_truth_labels = []
    for image_target_pred_element in pred_data:
        # NOTE: The boxes are already sorted in descending according to their confidence scores
        pred_for_one_image = image_target_pred_element['pred'] # get the model predictions for that image
        score = pred_for_one_image['scores'][0]
        scores_for_all_images.append(score) # get the maximum confidence score
        # Check if gt exists
        if image_target_pred_element['target']['boxes'].numel() == 0:
            ground_truth_labels.append(0)
        else:
            ground_truth_labels.append(1)

    
    scores_array = np.array(scores_for_all_images)
    is_malignant_predictions = scores_array > threshold # Malignant or benign by model
    matched = [] # stores if the prediction box has matched. if no gt, match = 0

    # Prepare the matched list. It stores 1 only if a malignant prediction has been made and if the box matches with gt
    # In all other cases it stores 0
    for img_indx, is_malignant in enumerate(is_malignant_predictions):
        # if labelled as malignant check if the box matches
        if is_malignant:
            # import ipdb; ipdb.set_trace()
            if pred_data[img_indx]['target']['boxes'].numel() == 0: # if cancer is not present and model predicts it somewhere, just append with 0
                matched.append(0)
                continue
            gt_box_coords = pred_data[img_indx]['target']['boxes'][0]
            # get highest confidence box from the model's prediction
            # import ipdb; ipdb.set_trace()
            pred_box_coords = pred_data[img_indx]['pred']['boxes'][0]
            # check if predicted box matches gt box
            does_pred_box_match = does_it_match_with_gt(gt_box_coords, pred_box_coords)
            if does_pred_box_match:
                matched.append(1)
            else:
                matched.append(0)
        else:
            matched.append(0)

    # Now measure D-ECE
    # 1) get the matched array. (already done above)
    # 2) get the confidences and relative_x_position and stack them up for binning
    
    # Step-1 : Matched arrray
    matched_array = np.array(matched)
    # Step-2 : get confidences and relative_x_position stacked up
    confidences = scores_array # 218 dimensional confidence vector
    relative_cx_position = np.array([
            img_target_pred_element['pred']['boxes'][0][0].tolist()
            for img_target_pred_element in pred_data
        ])
    relative_cy_position = np.array([
            img_target_pred_element['pred']['boxes'][0][1].tolist()
            for img_target_pred_element in pred_data
        ])
    relative_w_position = np.array([
            img_target_pred_element['pred']['boxes'][0][2].tolist()
            for img_target_pred_element in pred_data
        ])
    relative_h_position = np.array([
            img_target_pred_element['pred']['boxes'][0][3].tolist()
            for img_target_pred_element in pred_data
        ])
    
    # if n_bins.__len__() != 1:
    #     binning_axes = np.stack((confidences,
    #                             relative_cx_position, 
    #                             relative_cy_position, 
    #                             relative_w_position, 
    #                             relative_h_position), axis=1)
    #     # binning_axes = np.stack((confidences), axis=1)
    
    # elif n_bins.__len__() == 1:
    #     binning_axes = confidences

    # binning_axes = np.stack((confidences), axis=1)
    # binning_axes = confidences
    # dece_value = ece.measure(binning_axes, matched_array)

    ## CALCULATING AUC-Score # 1) Get a predicted_scores list for every image
    # 2) Get a ground truth list
    from sklearn.metrics import roc_auc_score

    # Step-1: Predicted Scores
    predicted_scores = confidences

    # Step-2: Ground Truths List
    ground_truth_labels = ground_truth_labels
    
    auc_score = roc_auc_score(ground_truth_labels, predicted_scores)
    
    # import ipdb; ipdb.set_trace()
    return auc_score


def get_preds_classification(dataset, args):
    data = pd.read_csv(dataset, header=None)
    # import pdb; pdb.set_trace()
    print(len(data))
    preds = []
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    postprocessors['bbox'].nms_iou_threshold = 0.25

    for i,row in ( tqdm(data.iterrows()) ):
            item_info = {}
            image_path, label = row
            image_array = Image.open(image_path).convert("RGB") 
            image, _ = transform(image_array, None)

            output = model.cuda()(image[None].cuda())
            output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
            output['scores'] = output['scores'].cpu()
            output['labels'] = output['labels'].cpu()
            output['boxes'] = box_ops.box_xyxy_to_cxcywh(output['boxes']).cpu()
            item_info['label'] = label
            item_info['pred'] = output
            # import pdb; pdb.set_trace()
            preds.append(item_info)
    return preds


def calc_accuracy_classification(pred_data, threshold = None, num_thresh=1000):
    num_images = len(pred_data)
    if(threshold==None):
        thresholds = np.linspace(0,0.2,num_thresh)
    else:
        thresholds = [threshold]
    metrics = np.zeros((num_thresh, 2))
    class_reports = []
    target_names = ['ben', 'mal']
    #tp, tn, fp, fn
    for i, thresh_val in enumerate( tqdm(thresholds) ):
        preds = np.zeros(len(pred_data))
        labels = np.zeros(len(pred_data))
        pos_idx = [i for i,item in enumerate(pred_data) if torch.max(item['pred']['scores'])>thresh_val]
        preds[pos_idx] = 1
        labels = [item['label'] for i,item in enumerate(pred_data)]
        conf_mat = np.array(confusion_matrix(labels, preds).ravel())
        # print(conf_mat)
        pres = conf_mat[3]/(conf_mat[3]+conf_mat[1]+ 1) + 0.0001
        recall = conf_mat[3]/(conf_mat[2]+conf_mat[3]+ 1) + 0.0001
        metrics[i,0] = 2*pres*recall/(pres+recall)
        metrics[i,1] = (conf_mat[3]+conf_mat[0])/(conf_mat[3]+conf_mat[1]+conf_mat[0]+conf_mat[2])
        class_reports.append(classification_report(labels, preds, target_names=target_names))
        # if(thresh_val>0.028 and thresh_val < 0.032):
        #     print("Threshold:", thresh_val)
        #     print("F1 score:", 2*pres*recall/(pres+recall))
        #     print("Accuracy:", (conf_mat[0]+conf_mat[1])/(conf_mat[0]+conf_mat[1]+conf_mat[2]+conf_mat[3]))
    max_f1, max_acc = np.argmax(metrics, axis=0)
    print("Max F1 score and Accuracy:", metrics[max_f1], "Threshold:", thresholds[max_f1])
    print(class_reports[max_f1])
    print("F1 score and Max Accuracy:", metrics[max_acc], "Threshold:", thresholds[max_acc])


def get_confmat_clf(pred_list, threshold=0.1):
    #tp, tn, fp, fn
    conf_mat = np.zeros((4))
    conf_mat_idx = []
    for i, data_item in enumerate(pred_list):
        gt_data = data_item['target']
        pred = data_item['pred']
        scores = pred['scores']
        select_mask = scores > threshold
        pred_boxes = pred['boxes'][select_mask]
        out_array = np.zeros((4))
        if(len(gt_data['boxes'])!=0 and len(pred_boxes)!=0):
            out_array[0]+=1
        elif(len(gt_data['boxes'])==0 and len(pred_boxes)!=0):
            out_array[2]+=1
        elif(len(gt_data['boxes'])!=0 and len(pred_boxes)==0):
            out_array[3]+=1
        else:
            out_array[1]+=1
        conf_mat+=out_array
    return conf_mat

def get_confmat(pred_list, threshold = 0.3):
    def true_positive(gt, pred):
        # If center of pred is inside the gt, it is a true positive
        box_pascal_gt = ( gt[0]-(gt[2]/2.) , gt[1]-(gt[3]/2.), gt[0]+(gt[2]/2.), gt[1]+(gt[3]/2.) )
        if (pred[0] >= box_pascal_gt[0] and pred[0] <= box_pascal_gt[2] and
                pred[1] >= box_pascal_gt[1] and pred[1] <= box_pascal_gt[3]):
            return True
        return False

    #tp, tn, fp, fn
    conf_mat = np.zeros((4))
    error_image = np.zeros((len(pred_list)))
    conf_mat_idx = []
    for i, data_item in enumerate(pred_list):
        gt_data = data_item['target']
        pred = data_item['pred']
        scores = pred['scores']
        select_mask = scores > threshold
        pred_boxes = pred['boxes'][select_mask]
        out_array = np.zeros((4))
        # if(i in [25,26,36,70,85]):
        
        for j, gt_box in enumerate(gt_data['boxes']):
            add_tp = False
            new_preds = []
            for pred in pred_boxes:
                # print(i,j)
                # if(i==175):
                #     import pdb; pdb.set_trace()
                if true_positive(gt_box, pred):
                    add_tp = True
                else:
                    new_preds.append(pred)
            pred_boxes = new_preds
            if add_tp:
                out_array[0] += 1
            else:
                out_array[3] += 1
        out_array[2] = len(pred_boxes)
        conf_mat+=out_array
        conf_mat_idx.append(out_array)
        if(out_array[0]!=0):
            error_image[i] = 1
        # if(out_array[2]!=0 or out_array[3]!=0):?d_ece

        #     error_image[i] = 1
    return conf_mat, error_image, conf_mat_idx


def get_certainty_confmat(pred_list, threshold = 0.3, conf_th=0.5):
    # print("conf_th",conf_th)

    def tp_conf_or_not(gt, pred, conf_th=0.5):
        conf_threshold = conf_th
        # If center of pred is inside the gt, it is a true positive
        box_pascal_gt = ( gt[0]-(gt[2]/2.) , gt[1]-(gt[3]/2.), gt[0]+(gt[2]/2.), gt[1]+(gt[3]/2.) )
        if (pred[0] >= box_pascal_gt[0] and pred[0] <= box_pascal_gt[2] and
                pred[1] >= box_pascal_gt[1] and pred[1] <= box_pascal_gt[3]):
            if pred[0] >= conf_threshold:
                return "TP_confident"
            else:
                return "TP_underconfident"

    def fp_conf_or_not(gt, pred, conf_th=0.5):
        conf_threshold = conf_th
        # If center of pred is inside the gt, it is a true positive
        box_pascal_gt = ( gt[0]-(gt[2]/2.) , gt[1]-(gt[3]/2.), gt[0]+(gt[2]/2.), gt[1]+(gt[3]/2.) )
        # if box doesn't match with gt
        if not ((pred[0] >= box_pascal_gt[0] and pred[0] <= box_pascal_gt[2] and
                pred[1] >= box_pascal_gt[1] and pred[1] <= box_pascal_gt[3])):
            if pred[0] >= conf_threshold:
                return "FP_confident"
            else:
                return "FP_underconfident"

    # import ipdb; ipdb.set_trace()
    #tp_conf, fp_underconf, tp_underconf, fp_conf
    conf_mat = np.zeros((4))
    error_image = np.zeros((len(pred_list)))
    conf_mat_idx = []
    for i, data_item in enumerate(pred_list):
        gt_data = data_item['target']
        pred = data_item['pred']
        scores = pred['scores']
        select_mask = scores > threshold
        pred_boxes = pred['boxes'][select_mask]
        out_array = np.zeros((4))
        # if(i in [25,26,36,70,85]):
        
        for j, gt_box in enumerate(gt_data['boxes']):
            add_tp_conf = False
            add_tp_underconf = False
            add_fp_conf = False
            add_fp_underconf = False
            # new_preds = []
            for pred in pred_boxes:
                # print(i,j)
                # if(i==175):
                #     import pdb; pdb.set_trace()
                if tp_conf_or_not(gt_box, pred, conf_th=conf_th) == "TP_confident":
                    add_tp_conf = True
                elif tp_conf_or_not(gt_box, pred, conf_th=conf_th) == "TP_underconfident":
                    add_tp_underconf = True
                elif fp_conf_or_not(gt_box, pred, conf_th=conf_th) == "FP_confident":
                    add_fp_conf = True
                elif fp_conf_or_not(gt_box, pred, conf_th=conf_th) == "FP_underconfident":
                    add_fp_underconf = True

            if add_tp_conf:
                out_array[0] += 1
            elif add_fp_underconf:
                out_array[1] += 1
            elif add_tp_underconf:
                out_array[2] += 1
            elif add_fp_conf:
                out_array[3] += 1
        # out_array[2] = len(pred_boxes)
        conf_mat+=out_array
        conf_mat_idx.append(out_array)
        # if(out_array[0]!=0):
        #     error_image[i] = 1
        # if(out_array[2]!=0 or out_array[3]!=0):?d_ece

        #     error_image[i] = 1
    return conf_mat, conf_mat_idx


def get_error_boxes(pred_data, save_folder, thresh = 0.1313131313):
    os.makedirs(save_folder, exist_ok=True)
    _, error_image, conf_mats = get_confmat(pred_data, thresh)
    error_idxs = np.where(error_image!=0)
    selected_pred = np.array(pred_data)[error_idxs[0]]
    selected_conf = np.array(conf_mats)[error_idxs[0]]
    # selected_pred = np.array(pred_data)
    vslzr = COCOVisualizer()
    for i, item in tqdm(enumerate(selected_pred)):
        # import pdb; pdb.set_trace()
        gt_data = item['target']
        pred = item['pred']
        scores = pred['scores']
        select_mask = scores > thresh
        box_labels = ["pred" for item in pred['boxes'][select_mask]] + ["gt" for item in gt_data['boxes']]
        pred_dict = {
            'boxes': torch.cat((pred['boxes'][select_mask],gt_data['boxes'])),
            'size': gt_data['size'],
            'box_label': box_labels,
            'image_id': gt_data['image_id']
        }
        vslzr.visualize(item['image'], pred_dict, savedir=save_folder, show_in_console=False)
    

def save_plot(senses, fps, data="l"):
    plt.plot(fps, senses)
    baselines = []
    if(data=='inbreast'):
        baselines.append(["o", 0.85, 0.88, "Siamese Faster-RCNN"])
        baselines.append(["v", 5, 0.95, "RCNN Dhungal"])
        baselines.append(["^", 0.3, 0.9, "Ribli"])
        baselines.append(["s", 0.58, 0.93, "CNN Aggarwal"])
        baselines.append(["p", 0.3, 0.92, "Richa"])
    elif(data=="ddsm"):
        baselines.append(["o", 1.9, 0.88, "CVR RCNN"])
        baselines.append(["v", 2.1, 0.85, "Faster-RCNN"])
        baselines.append(["^", 2.7, 0.88, "Sampat"])
        baselines.append(["s", 2.4, 0.88, "Eltonsy"])
        baselines.append(["p", 1, 0.78, "CBN"])
        baselines.append(["P", 1, 0.85, "MommiNet"])
        baselines.append(["*", 1.9, 0.92, "BG RCNN"])
        baselines.append(["d", 1.1, 0.8, "Campanini"])
        
    for [marker, x, y, label] in baselines:
        plt.scatter(x, y, marker = marker, label = label)
    plt.xlabel("False positives per image")
    plt.ylabel("Sensitivity")
    plt.title("{} FROC".format(data))
    plt.legend()
    plt.grid(True)
    plt.savefig("{}_froc_plot.png".format(data))
    plt.clf()

def calc_accuracy(pred_data, num_thresh=100):
    num_images = len(pred_data)
    thresholds = np.linspace(0,1,num_thresh)
    metrics = np.zeros((num_thresh, 2))

    #tp, tn, fp, fn
    for i, thresh_val in enumerate( tqdm(thresholds) ):
        conf_mat= get_confmat_clf(pred_data, thresh_val)
        pres = conf_mat[0]/(conf_mat[0]+conf_mat[2]+ 1) + 0.0001
        recall = conf_mat[0]/(conf_mat[0]+conf_mat[3]+ 1) + 0.0001
        metrics[i,0] = 2*pres*recall/(pres+recall)
        metrics[i,1] = (conf_mat[0]+conf_mat[1])/(conf_mat[0]+conf_mat[1]+conf_mat[2]+conf_mat[3])
        if(thresh_val>0.028 and thresh_val < 0.032):
            print("Threshold:", thresh_val)
            print("F1 score:", 2*pres*recall/(pres+recall))
            print("Accuracy:", (conf_mat[0]+conf_mat[1])/(conf_mat[0]+conf_mat[1]+conf_mat[2]+conf_mat[3]))
    max_f1, max_acc = np.argmax(metrics, axis=0)
    print("Max F1 score and Accuracy:", metrics[max_f1], "Threshold:", thresholds[max_f1])
    print("F1 score and Max Accuracy:", metrics[max_acc], "Threshold:", thresholds[max_acc])
    
    
    
# fps_req: false positives required
def calc_froc(pred_data, fps_req = [0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,1.4,1.6,2,3,4,5], num_thresh = 1000):
    # import ipdb; ipdb.set_trace()
    num_images = len(pred_data)
    thresholds = np.linspace(0,1,num_thresh)
    conf_mat_thresh = np.zeros((num_thresh, 4))
    # certainty_conf_mat_thresh = np.zeros((num_thresh, 4))

    for i, thresh_val in enumerate( tqdm(thresholds) ):
        conf_mat,_,_ = get_confmat(pred_data, thresh_val) # confusion matrix:
                                                            # [TP, TN, FP, FN]
        conf_mat_thresh[i] = conf_mat
        # certainty_conf_mat, _ = get_certainty_confmat(pred_data, thresh_val, conf_th=0.25)
        # certainty_conf_mat_thresh[i] = certainty_conf_mat
    
    # certainty_conf_mat_cert_th = np.zeros((10, num_thresh, 4))
    # _certainty_conf_mat_thresh = np.zeros((num_thresh, 4))
    # c_tpr_list = []
    # _c_tpr = np.zeros((num_thresh))
    # for j, certainty_th in enumerate(tqdm(np.linspace(0, 1, 10))):
    # for i, thresh_val in enumerate(tqdm(thresholds)):
    #     certainty_conf_mat, _ = get_certainty_confmat(pred_data, thresh_val, conf_th=0.25)
    #     certainty_conf_mat_thresh[i] = certainty_conf_mat
    #     ccm = certainty_conf_mat_thresh[i]
    #     if ((ccm[0]+ccm[1]+ccm[2]+ccm[3]==0)):
    #         c_tpr[i] = 0
    #     else:
    #         c_tpr[i] = (ccm[0]+ccm[1]) / (ccm[0]+ccm[1]+ccm[2]+ccm[3])

    # import ipdb; ipdb.set_trace()
    sensitivity = np.zeros((num_thresh)) #recall
    specificity = np.zeros((num_thresh)) #precision
    # c_tpr = np.zeros((num_thresh))
    for i in range(num_thresh):
        conf_mat = conf_mat_thresh[i]
        if((conf_mat[0]+conf_mat[3])==0):
            sensitivity[i] = 0
        else:
            sensitivity[i] = conf_mat[0]/(conf_mat[0]+conf_mat[3])
        if((conf_mat[0]+conf_mat[2])==0):
            specificity[i] = 0
        else:
            specificity[i] = conf_mat[0]/(conf_mat[0]+conf_mat[2])
    
        # ccm = certainty_conf_mat_thresh[i]
        # if ((ccm[0]+ccm[1]+ccm[2]+ccm[3]==0)):
        #     c_tpr[i] = 0
        # else:
        #     c_tpr[i] = (ccm[0]+ccm[1]) / (ccm[0]+ccm[1]+ccm[2]+ccm[3])


    senses_req = [] # sensitivity required
    c_tpr_req = []
    d_ece_conf_values = [] # d_ece_values corresponding to the thresholds
    d_ece_mulbins_values = [] # d_ece_values corresponding to the thresholds
    auc_values = []
    print('False Positives Required  |  Sensitivity  |  Thresholds  | AUC Score (computed over complete curve)')
    for fp_req in fps_req:
        for i in range(num_thresh):
            f = conf_mat_thresh[i][2] # number of false_positives
            if f/num_images < fp_req:
                senses_req.append(sensitivity[i - 1])
                # c_tpr_req.append(c_tpr[i - 1])
                c_tpr_req.append(0)
                # import ipdb; ipdb.set_trace()
                auc_score = get_D_ECE_and_auc(pred_data, threshold=thresholds[i], n_bins=[10])
                # dece_mulbins, auc_score = get_D_ECE_and_auc(pred_data, threshold=thresholds[i], n_bins=[10,10,10,10,10])
                # d_ece_conf_values.append(dece_conf)
                # d_ece_mulbins_values.append(dece_mulbins)
                auc_values.append(auc_score)
                # print(fp_req, sensitivity[i - 1], thresholds[i], dece_conf, dece_mulbins, auc_score)
                print(fp_req, sensitivity[i - 1], thresholds[i], auc_score)
                break
    # import ipdb; ipdb.set_trace()
    save_plot(senses_req, fps_req, data="ddsm")
    # return senses_req, c_tpr_req, fps_req, sensitivity, specificity, d_ece_conf_values, d_ece_mulbins_values, auc_values
    return senses_req, c_tpr_req, fps_req, sensitivity, specificity, auc_values

if __name__ == '__main__':
  
    ROOT = "/home/tajamul/scratch/DA/Checkpoints/source_only/Focalnet/aiims"




    #path where weights are stored
    # ROOT = "/home/kshitiz/scratch/FocalNet-DINO/exps/inbreast_aiims_2k"
    # checkpoint_number = 40
    # checkpoint_number = 25
    for checkpoint_number in range(1, 21):
        print(f"Processing checkpoint {checkpoint_number}...")  
        model_config_path = os.path.join(ROOT,"config_cfg.py") # change the path of the model config file
        model_checkpoint_path = os.path.join(ROOT,"checkpoint00{}.pth".format(checkpoint_number)) # change the path of the model checkpoint
        # model_checkpoint_path = os.path.join(ROOT,"teacher_checkpoint000{}.pth".format(checkpoint_number)) # change the path of the model checkpoint
        # model_checkpoint_path = os.path.join(ROOT,"teacher_checkpoint000{}.pth".format(checkpoint_number)) # change the path of the model checkpoint
        # model_checkpoint_path = os.path.join(ROOT,"checkpoint_best_regular.pth") # change the path of the model checkpoint
        
        # parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
        # args = parser.parse_args()
        args = SLConfig.fromfile(model_config_path) 
        args.device = 'cuda'
        args.lambda_eckd = 1.0
        args.temp = 1.0
        args.memory_bank_size = 1
        model, criterion, postprocessors = build_model_main(args)
    

        model_checkpoint = torch.load(model_checkpoint_path, map_location=args.device)
        model.load_state_dict(model_checkpoint['model'])

        _ = model.eval()
        args.fix_size = True 

        

        args.dataset_file = 'coco'

        args.coco_path = "/home/tajamul/scratch/DA/DATA/DATA/Coco_Data/BCD/INBreast"
        # args.coco_path = "/home/kartik_anand/scratch/kstyles/miccai/cal-detr/data/DDSM" # the path of coco
        
        #import ipdb; ipdb.set_trace()
        # dataset_val = build_dataset(image_set='val', args=args)   
        # dataset_train = build_dataset(image_set='train', args=args)
        dataset_test = build_dataset(image_set='test', args=args)

        # print(f"Dataset_items: Train: {dataset_train.__len__()} | Test: {dataset_test.__len__()}")
        print(f"Test: {dataset_test.__len__()}")

        pred_list_test, model_out_test = get_preds(dataset_test)
        # import ipdb; ipdb.set_trace()
        # senses_req, fps_req, sensitivity, specificity, d_ece_conf_values, d_ece_mulbins_values, auc_values
        v_froc, v_fpi, v_recall, v_pres, v_dece, v_dece_mulbins, v_auc = calc_froc(pred_list_test)
        # v_froc, v_fpi, v_recall, v_pres, v_auc = calc_froc(pred_list_test)
        print(v_froc, v_fpi, v_recall, v_pres)
        calc_accuracy(pred_list_test)
