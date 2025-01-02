import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model_vitb16_dino import vitdino
from data_single import SingleImageDataset
    


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def load_model_again(checkpoint_path, layers_freeze, img_size):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(device)

    model =vitdino(layers_freeze, img_size)
    model.load_state_dict(remove_module_prefix(torch.load(checkpoint_path)))
    model.to(device)

    model = torch.nn.DataParallel(model)

    return model

def test_code(model, test_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    predictions = []
    true_labels = []
    prob_val = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            crops = batch 
            crops = crops.to(device)
            logits = model(crops)

            probabilities = F.softmax(logits, dim=-1)
            pred = probabilities.max(1, keepdim=True)[1]
            greater_prob = [x[1] for x in probabilities.tolist()]

            predictions.extend([x[0] for x in pred.tolist()])
            prob_val.extend(greater_prob)

            # Assuming batch contains images and labels
            images = batch
            for image, pred_label in zip(images, pred):
                label_text = "cancerous" if pred_label.item() == 1 else "non-cancerous"
                print(f"Predicted Label: {label_text}")
                # You can print additional information here if needed

 

if __name__=="__main__":
    
    

    # checkpoint_path = "./result_models/rn50_imgnet/model_best.pt"
    checkpoint_path = "./models/vitdino/aiims/model_best.pt"
    num_workers = 8
    batch_size = 4
    topk = 8
    img_size = 224

    layers_freeze = 2

    print(f'topk = {topk}\nnum_workers = {num_workers}\nbatch_size = {batch_size}\nimage = {img_size}\nlayers_freeze = {layers_freeze}')

    model = load_model_again(checkpoint_path, layers_freeze, img_size)
    
    
    print("Loading validation DataLoader: ")
    
    text_path = "/home/tajamul/scratch/DA/Datasets/Foundation_Data/INBreast/test2017_focalnet_a25/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm_preds.txt"
    image_path = "/home/tajamul/scratch/DA/Datasets/Foundation_Data/INBreast/test2017/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm.png"
    
    dataset = SingleImageDataset(image_path, text_path, topk)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)   
    print("Now Testing: ")
    test_code(model, data_loader)

