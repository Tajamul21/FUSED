# import numpy as np
# import pandas as pd
# import os
# from sklearn.metrics import f1_score
# from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# def create_label_map(gt_file, pred_file):
#     gt_df = pd.read_csv(gt_file)
#     pred_df = pd.read_csv(pred_file)
#     try:
#         merged = gt_df.merge(pred_df, on='img_path', how='inner')
#     except:
#         import ipdb; ipdb.set_trace()
    
#     mal_logits = merged['mal_score'].to_numpy()
#     true_labels = merged['label'].to_numpy()
#     return true_labels, mal_logits

# def fpr2recall(logits, labels, fpr_value = 0.3):
#     # Calculate threshold based on fixed FPR
#     neg_idx = np.where(np.array(labels) == 0)[0]
#     neg_logits = sorted([logits[idx] for idx in neg_idx], reverse=True)
#     if fpr_value == 1.0:
#         thresh = neg_logits[int(fpr_value * len(neg_logits))-1]
#     else:
#         thresh = neg_logits[int(fpr_value * len(neg_logits))]
    
#     # Make predictions based on threshold
#     logits = np.array(logits)
#     preds = np.zeros(len(labels))
#     preds[np.where(logits >= thresh)[0]] = 1
    
#     # Calculate recall (TPR)
#     true_positives = sum(1 for pred, gold in zip(preds, labels) if pred == 1 and gold == 1)
#     actual_positives = sum(labels)
#     tpr = true_positives / actual_positives if actual_positives != 0 else 0
    
#     # Calculate F1 score
#     f1 = f1_score(labels, preds)

#     return tpr, thresh, f1
    
# def calc_thresh(true_labels, mal_logits):
#     fprs = np.linspace(0, 1.0, num=1000)
#     # for i, fpr in enumerate(fprs):
#     #     tpr, thresh, f1 = fpr2recall(mal_logits, true_labels, fpr_value=fpr)
#     #     print(f'{fpr}, {tpr}, {thresh}, {f1}')

#     tpr, thresh, f1 = [], [], []

#     for fpr_val in fprs: 
#         tpr_in, thresh_in, f1_in = fpr2recall(mal_logits, true_labels, fpr_val)
#         tpr.append(tpr_in), thresh.append(thresh_in), f1.append(f1_in)

#     results_df = pd.DataFrame({
#         'FPR': fprs,
#         'TPR': tpr,
#         'Threshold': thresh,
#         'F1 Score': f1
#     })

#     return results_df
    
# if __name__=='__main__':
#     model_name = "fnd_source"
#     # pred_file = f"./{model_name}_preds.csv"
#     pred_file = "/home/kartik_anand/scratch/kstyles/miccai/SFDA/Proposed/FocalNet_DINO/fndsource_cview_preds.csv"
#     gt_file = "/home/kartik_anand/scratch/kstyles/miccai/IRCH/annotations/val_gt.csv"

#     true_labels, mal_logits = create_label_map(gt_file, pred_file)
#     # calc_metrics(true_labels, mal_logits, threshold, model_name+"_metrics.txt")
#     results_df = calc_thresh(true_labels, mal_logits)
#     results_df.to_csv(f'{model_name}_thresh.csv')


import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def create_label_map(gt_file, pred_file):
    gt_df = pd.read_csv(gt_file)
    pred_df = pd.read_csv(pred_file)
    merged = gt_df.merge(pred_df, on='img_path', how='inner')
    
    mal_logits = merged['mal_score'].to_numpy()
    true_labels = merged['label'].to_numpy()
    return true_labels, mal_logits

def fpr2recall(logits, labels, fpr_value = 0.3):
    # Calculate threshold based on fixed FPR
    neg_idx = np.where(np.array(labels) == 0)[0]
    neg_logits = sorted([logits[idx] for idx in neg_idx], reverse=True)
    if fpr_value == 1.0:
        thresh = neg_logits[int(fpr_value * len(neg_logits))-1]
    else:
        thresh = neg_logits[int(fpr_value * len(neg_logits))]
    
    # Make predictions based on threshold
    logits = np.array(logits)
    preds = np.zeros(len(labels))
    preds[np.where(logits >= thresh)[0]] = 1
    
    # Calculate recall (TPR)
    true_positives = sum(1 for pred, gold in zip(preds, labels) if pred == 1 and gold == 1)
    actual_positives = sum(labels)
    tpr = true_positives / actual_positives if actual_positives != 0 else 0
    
    # Calculate F1 score
    f1 = f1_score(labels, preds)

    return tpr, thresh, f1
    
def calc_thresh(true_labels, mal_logits):
    fprs = np.linspace(0, 1.0, num=1000)
    # for i, fpr in enumerate(fprs):
    #     tpr, thresh, f1 = fpr2recall(mal_logits, true_labels, fpr_value=fpr)
    #     print(f'{fpr}, {tpr}, {thresh}, {f1}')

    tpr, thresh, f1 = [], [], []

    for fpr_val in fprs: 
        tpr_in, thresh_in, f1_in = fpr2recall(mal_logits, true_labels, fpr_val)
        tpr.append(tpr_in), thresh.append(thresh_in), f1.append(f1_in)

    results_df = pd.DataFrame({
        'FPR': fprs,
        'TPR': tpr,
        'Threshold': thresh,
        'F1 Score': f1
    })

    return results_df
    
if __name__=='__main__':
    model_name = "fnd_adapt_ckpt0003"
    # model_name = "fnd_adapt_ckpt0001"
    # pred_file = f"./{model_name}_preds.csv"
    # FOR SOURCE ONLY (AIIMS)
    # pred_file = "/home/kartik_anand/scratch/kstyles/miccai/IRCH/annotations/preds_fnd_source/fnd_source_preds_val.csv"
    # gt_file = "/home/kartik_anand/scratch/kstyles/miccai/IRCH/annotations/preds_fnd_source/val_gt.csv"
    
    # FOR SOURCE ONLY (GBCNet)
    # pred_file = "/home/kartik_anand/scratch/kstyles/miccai/SFDA/Proposed/FocalNet_DINO/exps/source_GBCNet/target_GBC_center/preds_source/fnd_sourceonly_ckpt24.csv"
    # ADAPT (GBC_center)
    # pred_file = "/home/kartik_anand/scratch/kstyles/miccai/SFDA/Proposed/FocalNet_DINO/exps/source_GBCNet/target_GBC_center/preds_source/fnd_adapt_ckpt0001_preds.csv"
    # pred_file = "/home/kartik_anand/scratch/kstyles/miccai/SFDA/Proposed/FocalNet_DINO/exps/source_GBCNet/target_GBC_center/preds_source/fnd_adapt_ckpt0002_preds.csv"
    pred_file = "/home/kartik_anand/scratch/kstyles/miccai/SFDA/Proposed/FocalNet_DINO/exps/source_GBCNet/target_GBC_center/preds_source/fnd_adapt_ckpt0003_preds.csv"
    # pred_file = "/home/kartik_anand/scratch/kstyles/miccai/SFDA/Proposed/FocalNet_DINO/exps/source_GBCNet/target_GBC_center/preds_source/fnd_adapt_ckpt0000_preds.csv"
    gt_file = "/home/kartik_anand/scratch/kstyles/miccai/SFDA/Proposed/FocalNet_DINO/exps/source_GBCNet/target_GBC_center/preds_source/test.csv"

    # FOR ADAPT (C-VIEW)
    # pred_file = "/home/kartik_anand/scratch/kstyles/miccai/IRCH/annotations/preds_fnd_source/fnd_source_preds_val.csv"
    # gt_file = "/home/kartik_anand/scratch/kstyles/miccai/IRCH/annotations/preds_fnd_source/val_gt.csv"
    # pred_file = "/home/kartik_anand/scratch/kstyles/miccai/SFDA/Proposed/FocalNet_DINO/cview_preds/fndadapt_sfdaexp_ckpt0002.csv"
    # gt_file = "/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/BCD/c_view_data/irch_gt.csv"

    true_labels, mal_logits = create_label_map(gt_file, pred_file)
    # calc_metrics(true_labels, mal_logits, threshold, model_name+"_metrics.txt")
    # import ipdb; ipdb.set_trace()
    results_df = calc_thresh(true_labels, mal_logits)
    results_df.to_csv(f'/home/kartik_anand/scratch/kstyles/miccai/SFDA/Proposed/FocalNet_DINO/exps/source_GBCNet/target_GBC_center/preds_source/{model_name}_thresh.csv')