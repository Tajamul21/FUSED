import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from args import get_args
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from model_vitb16_clip import CLIP_IMAGE
from model_vitb16_dino import vitdino
from model_vitb16_imgnet import vit_b_16
from data_da import all_mammo_da
import numpy as np

    
def load_data(CSV, IMG_BASE, TEXT_BASE, workers=8, batch_size=32, topk=5, img_size=224):
    dataset = all_mammo_da(CSV, IMG_BASE, TEXT_BASE, topk=topk, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers) 

    return dataloader

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# def load_model_again(checkpoint_path, layers_freeze, img_size):
#     device = "cuda" if torch.cuda.is_available() else "cpu" 
#     print(device)

#     model = vitb16_imgnet(layers_freeze, img_size)
#     model.load_state_dict(remove_module_prefix(torch.load(checkpoint_path)))
#     model.to(device)

#     model = torch.nn.DataParallel(model)

#     return model

def load_model_vitb16clip(checkpoint_path, class_num, layers_freeze, img_size):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(device)

    model = CLIP_IMAGE(layers_freeze, class_num=class_num)
    model.load_state_dict(remove_module_prefix(torch.load(checkpoint_path)))
    model.to(device)

    model = torch.nn.DataParallel(model)
#
    return model

def load_model_vitdino(checkpoint_path, class_num, layers_freeze, img_size):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(device)

    model = vitdino(class_num=class_num, layers=layers_freeze, img_size=img_size)
    model.to(device)

    model = torch.nn.DataParallel(model)
#
    return model

def load_model_vitb16imgnet(checkpoint_path, class_num, layers_freeze, img_size):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(device)

    model = vit_b_16(class_num=class_num, layers_freeze=layers_freeze, img_size=img_size)
    model.to(device)

    model = torch.nn.DataParallel(model)
#
    return model

def tsne(embedding_values_tensor, labels_tensor, plot_save):
    embedding_values = embedding_values_tensor.cpu().numpy()
    labels = labels_tensor.cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    embedded_values = tsne.fit_transform(embedding_values)

    plt.figure(figsize=(10, 8))

    for class_label in np.unique(labels):
        indices = labels == class_label
        label_name = 'Malignant' if class_label == 1 else 'Benign'
        plt.scatter(embedded_values[indices, 0], embedded_values[indices, 1], label=label_name)


    # plt.title('focal_crops_VITDINO aiims t-SNE Plot of Embedding Values')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig(plot_save[:-7] + "tsne.png")
    plt.clf()

import ast

def test_code(model, test_dataloader, plot_path, file_path, logits_save, args):
    file = open(file_path, "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    predictions = []
    true_labels = []
    prob_val = []
    logits_labels = np.empty((0, 3)) 
    if args.model_type == "vitdino":
        embeddings = torch.empty((0, 384), dtype=torch.float32) #vitdino
    elif args.model_type == "vitb16_clip":
        embeddings = torch.empty((0, 512), dtype=torch.float32) #vitb16_clip
    elif args.model_type == "vitb16_imgnet":
        embeddings = torch.empty((0, 768), dtype=torch.float32) #vitb16_imgnet
        
    # embeddings = torch.empty((0, 768), dtype=torch.float32)   #vitb16
    labels_tensor = torch.empty((0), dtype=torch.int)
    embeddings = embeddings.to(device)
    labels_tensor = labels_tensor.to(device)

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            crops, labels = batch 
            
            labels = torch.tensor([ast.literal_eval(s) for s in labels])

            crops = crops.to(device)
            labels = labels.to(device)

            logits, features = model(crops)
            logits = logits[0]

            # import pdb; pdb.set_trace()
            probabilities = F.softmax(logits, dim=-1)
            features = features.to(device)
            probs = probabilities.tolist()
            labs = labels.tolist()
            # import ipdb; ipdb.set_trace()
            # for index, prob in enumerate(probs):
            #     prob.append(labs[index])
            
            embeddings = torch.cat((embeddings, features), 0)
            labels_tensor = torch.cat((labels_tensor, labels), 0)
            # logits_labels = np.vstack([logits_labels, probs])
            
            
            # pred = probabilities.max(1, keepdim=True)[1]
            # greater_prob = [x[1] for x in probabilities.tolist()]
            
            # predictions.extend([x[0] for x in pred.tolist()])
            # true_labels.extend(labels.tolist())
            
            # prob_val.extend(greater_prob)
            
        # np.save(logits_save, logits_labels)
        np.save(logits_save[:-17] + "embeddings_save.npy", embeddings.cpu().numpy())
        # tsne(embeddings, labels_tensor, plot_path)
        
            
        # accuracy = accuracy_score(true_labels, predictions)
        # f1 = f1_score(true_labels, predictions)
        # print(f'Accuracy: {accuracy:.2f}')
        # print(f'F1 Score: {f1:.2f}')
        # file.write(f'Accuracy: {accuracy:.2f}\n')
        # file.write(f'F1 Score: {f1:.2f}\n')

        # print(classification_report(true_labels, predictions, labels=[0, 1]))
        # file.write(classification_report(true_labels, predictions, labels=[0, 1]))

        # auc_score = roc_auc_score(true_labels, prob_val)
        # print('Logistic: ROC AUC=%.3f' % (auc_score))
        # file.write('Logistic: ROC AUC=%.3f\n' % (auc_score))
        # # calculate roc curves
        # lr_fpr, lr_tpr, _ = roc_curve(true_labels, prob_val) 
        # plt.plot(lr_fpr, lr_tpr, marker='.', label='text')

        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.legend()
        # plt.savefig(plot_path)
        # plt.show()

if __name__=="__main__":
    ################### SET THESE VARIABLES FIRST ACCORDINGLY #######################
    # Target dataset: INBreast
    TEST_CSV = "/home/keerti_km/scratch/Negroni_datasets/Coco_Data/BCD/INBreast/test_id.csv"
    TEST_IMG_BASE = "/home/keerti_km/scratch/Negroni_datasets/Coco_Data/BCD/INBreast/test2017"
    TEST_FND_CROPS = "/home/keerti_km/scratch/Negroni_datasets/Coco_Data/BCD/INBreast/test2017_focalnet_crops"
    #################################################################################
    
    args = get_args()

    checkpoint_path = f"../foundation_weights/{args.model_type}/{args.dataset_source}/model_best.pt"
    # checkpoint_path = "./result_models/rn50_imgnet/model_best.pt"
    # checkpoint_path = "/home/kartik_anand/scratch/kstyles/miccai/SFDA/Foundation_Models/focalnet_crops/foundation_models/{args.model_type}/{args.dataset_source}/model_best.pt"
    # checkpoint_path = f"/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_CHECKPOINTS/ablations/topk_foundation_kitti2city/topk_{args.topk}/{args.dataset_source}_trained/{args.model_type}/model_best.pt"
    
    num_workers = 8
    batch_size = 1
    
    # plot_path = f'/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_CHECKPOINTS/ablations/topk_foundation_kitti2city/topk_{args.topk}/embeddings_on_{args.dataset_target}/{args.model_type}/result_auc_plot.png'
    # score_file = f'/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_CHECKPOINTS/ablations/topk_foundation_kitti2city/topk_{args.topk}/embeddings_on_{args.dataset_target}/{args.model_type}/result_scores.txt'
    # logits_save = f'/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_CHECKPOINTS/ablations/topk_foundation_kitti2city/topk_{args.topk}/embeddings_on_{args.dataset_target}/{args.model_type}/logits_labels.npy'
    
    plot_path = f'../foundation_weights/{args.model_type}/{args.dataset_source}/results_{args.dataset_target}/result_auc_plot.png'
    score_file = f'../foundation_weights/{args.model_type}/{args.dataset_source}/results_{args.dataset_target}/result_scores.txt'
    logits_save = f'../foundation_weights/{args.model_type}/{args.dataset_source}/results_{args.dataset_target}/logits_labels.npy'
    topk = args.topk
    img_size = args.img_size

    layers_freeze = args.layers_freeze

    print(f'topk = {topk}\nnum_workers = {num_workers}\nbatch_size = {batch_size}\nimage = {img_size}\nlayers_freeze = {layers_freeze}')
    
    class_num = int(args.class_num)
    if args.model_type == "vitb16_clip":
        model = load_model_vitb16clip(checkpoint_path, class_num, layers_freeze, img_size)
    elif args.model_type == "vitdino":
        model = load_model_vitdino(checkpoint_path, class_num, layers_freeze, img_size)
    elif args.model_type == "vitb16_imgnet":
        model = load_model_vitb16imgnet(checkpoint_path, class_num, layers_freeze, img_size)

    # model = load_model_again(checkpoint_path, layers_freeze, img_size)
    print("Loading validation DataLoader: ")
    val_dataloader = load_data(TEST_CSV, TEST_IMG_BASE, TEST_FND_CROPS, num_workers, batch_size, topk, img_size)    
    print("Now Testing: ")
    test_code(model, val_dataloader, plot_path, score_file, logits_save, args)

