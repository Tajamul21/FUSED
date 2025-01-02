import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from model_vitb16_dino import vitdino
from model_vitb16_imgnet import vit_b_16
from model_vitb16_clip import CLIP_IMAGE
from data_da import all_mammo_da
import numpy as np
import pandas as pd

    
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

def load_vitdino(checkpoint_path, layers_freeze, img_size):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(device)

    model =vitdino(layers_freeze, img_size)
    model.load_state_dict(remove_module_prefix(torch.load(checkpoint_path)))
    model.to(device)

    model = torch.nn.DataParallel(model)

    return model

def load_imagenet(checkpoint_path, layers_freeze, img_size):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(device)

    model =vit_b_16(layers_freeze, img_size)
    model.load_state_dict(remove_module_prefix(torch.load(checkpoint_path)))
    model.to(device)

    model = torch.nn.DataParallel(model)

    return model

def load_clip(checkpoint_path, layers_freeze, img_size):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(device)

    model = CLIP_IMAGE(layers_freeze)
    model.load_state_dict(remove_module_prefix(torch.load(checkpoint_path)))
    model.to(device)

    model = torch.nn.DataParallel(model)

    return model


def test_vitdino(model, test_dataloader):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    predictions = []
    true_labels = []
    image_paths = []
    conf_scores = []
    

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            crops, labels, image_path = batch 
            # import pdb; pdb.set_trace()

            crops = crops.to(device)
            labels = labels.to(device)

            logits, _ = model(crops)

            # import pdb; pdb.set_trace()
            probabilities = F.softmax(logits, dim=-1)
            probs = probabilities.tolist()
            labs = labels.tolist()
            for index, prob in enumerate(probs):
                prob.append(labs[index])
              
            
            pred = probabilities.max(1, keepdim=True)[1]
            greater_prob = [x[1] for x in probabilities.tolist()]
            
            predictions.extend([x[0] for x in pred.tolist()])
            true_labels.extend(labels.tolist())
            image_paths.extend(image_path)
            
            conf_scores.extend(greater_prob)

            
            
    return conf_scores, predictions, true_labels, image_paths    
            

def test_imagenet(model, test_dataloader):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    predictions = []
    true_labels = []
    conf_scores = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            crops, labels, _ = batch 

            crops = crops.to(device)
            labels = labels.to(device)

            logits, _ = model(crops)

            # import pdb; pdb.set_trace()
            probabilities = F.softmax(logits, dim=-1)
            probs = probabilities.tolist()
            labs = labels.tolist()
            for index, prob in enumerate(probs):
                prob.append(labs[index])
              
            
            pred = probabilities.max(1, keepdim=True)[1]
            greater_prob = [x[1] for x in probabilities.tolist()]
            
            predictions.extend([x[0] for x in pred.tolist()])
            true_labels.extend(labels.tolist())
            
            conf_scores.extend(greater_prob)
            
            
    return conf_scores, predictions, true_labels             
            
    
def test_clip(model, test_dataloader):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    predictions = []
    true_labels = []
    conf_scores = []
    

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            crops, labels, _ = batch 

            crops = crops.to(device)
            labels = labels.to(device)

            logits, _ = model(crops)

            # import pdb; pdb.set_trace()
            probabilities = F.softmax(logits, dim=-1)
            probs = probabilities.tolist()
            labs = labels.tolist()
            for index, prob in enumerate(probs):
                prob.append(labs[index])
              
            
            pred = probabilities.max(1, keepdim=True)[1]
            greater_prob = [x[1] for x in probabilities.tolist()]
            
            predictions.extend([x[0] for x in pred.tolist()])
            true_labels.extend(labels.tolist())
            
            conf_scores.extend(greater_prob)
            
            
    return conf_scores, predictions, true_labels        


def calculate_uncertainty(predictions_vitdino, predictions_imagenet, predictions_clip):
    # Calculate uncertainty
    A_score_list = []
    for i in range(len(predictions_vitdino)):
        g = np.array([predictions_vitdino[i], predictions_imagenet[i], predictions_clip[i]]).reshape(-1, 1)
        g_mean = np.mean(g)
        uc = np.mean(np.sum(g ** 2)) - np.sum(g_mean ** 2)
        image_uncertainty = np.sum(uc)
        A_score_list.append(image_uncertainty)

    return A_score_list

if __name__=="__main__":


    TEST_CSV = "/home/kartik_anand/scratch/kstyles/miccai/Foundation_Data/INBreast/test.csv"
    TEST_IMG_BASE = "/home/kartik_anand/scratch/kstyles/miccai/Foundation_Data/INBreast/test2017"
    TEST_TEXT_BASE = "/home/kartik_anand/scratch/kstyles/miccai/Foundation_Data/INBreast/test2017_focalnet_a25"

    # checkpoint_path = "./result_models/rn50_imgnet/model_best.pt"
    checkpoint_path_vitdino = "/home/kartik_anand/scratch/kstyles/miccai/SFDA/Foundation_Models/focalnet_crops/models/vitdino/aiims/model_best.pt"
    checkpoint_path_imagenet = "/home/kartik_anand/scratch/kstyles/miccai/SFDA/Foundation_Models/focalnet_crops/models/vitb16_imgnet/aiims/model_best.pt"
    checkpoint_path_clip = "/home/kartik_anand/scratch/kstyles/miccai/SFDA/Foundation_Models/focalnet_crops/models/vit16_clip/aiims/model_best.pt"
    num_workers = 8
    batch_size = 1

    topk = 8
    img_size = 224

    layers_freeze = 2

    print(f'topk = {topk}\nnum_workers = {num_workers}\nbatch_size = {batch_size}\nimage = {img_size}\nlayers_freeze = {layers_freeze}')

    model_vitdino = load_vitdino(checkpoint_path_vitdino, layers_freeze, img_size)
    model_imagenet = load_imagenet(checkpoint_path_imagenet, layers_freeze, img_size)
    model_clip = load_clip(checkpoint_path_clip, layers_freeze, img_size)
    print("Loading validation DataLoader: ")
    val_dataloader = load_data(TEST_CSV, TEST_IMG_BASE, TEST_TEXT_BASE, num_workers, batch_size, topk, img_size)    
    print("Now Testing: ")
    conf_vitdino, _, true_labels, image_path = test_vitdino(model_vitdino, val_dataloader)
    conf_imagenet,_, _ = test_imagenet(model_imagenet, val_dataloader)
    conf_clip,_, _ = test_clip(model_clip, val_dataloader)
    
    predictions_vitdino = np.array(conf_vitdino)
    predictions_imagenet = np.array(conf_imagenet)
    predictions_clip = np.array(conf_clip)

    import ipdb; ipdb.set_trace()
    
    uncertainty = calculate_uncertainty(predictions_vitdino, predictions_imagenet, predictions_clip)
    print(uncertainty)

    # import pdb; pdb.set_trace()

    A_score_list = np.array(uncertainty)
    img_paths = np.array(image_path)

    # Sort the image paths based on uncertainty scores
    index = np.argsort(A_score_list)

    # Define your threshold for similarity
    pre_defined_threshold = 0.8  # Set your own threshold here
    num_similar = int((1 - pre_defined_threshold) * len(img_paths))

    # Load the main actual CSV file
    main_csv_path = "/home/tajamul/scratch/DA/Datasets/Foundation_Data/INBreast/test.csv"
    main_df = pd.read_csv(main_csv_path)

    # Get the paths for similar and dissimilar images
    img_paths_similar = img_paths[index][-num_similar:]
    img_paths_dissimilar = img_paths[index][:-num_similar]

    # Filter the main DataFrame based on image paths for similar and dissimilar images
    df_similar_main = main_df[main_df['im_path'].isin(img_paths_similar)]
    df_dissimilar_main = main_df[main_df['im_path'].isin(img_paths_dissimilar)]

    # Save DataFrames to CSV files
    similar_csv_path = "train_similar.csv"  # Path for similar CSV file
    df_similar_main.to_csv(similar_csv_path, index=False)

    dissimilar_csv_path = "train_dissimilar.csv"  # Path for dissimilar CSV file
    df_dissimilar_main.to_csv(dissimilar_csv_path, index=False)

    print("CSV files saved successfully.")