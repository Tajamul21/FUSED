import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model_vitb16_clip import CLIP_IMAGE
from data import all_mammo
    
def load_data(CSV, IMG_BASE, TEXT_BASE, workers=8, batch_size=32, topk=5, img_size=224):
    dataset = all_mammo(CSV, IMG_BASE, TEXT_BASE, topk=topk, img_size=img_size)
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

def load_model_again(checkpoint_path, layers_freeze, img_size):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(device)

    model = CLIP_IMAGE(layers_freeze)
    model.load_state_dict(remove_module_prefix(torch.load(checkpoint_path)))
    model.to(device)

    model = torch.nn.DataParallel(model)

    return model

def test_code(model, test_dataloader, plot_path, file_path):
    file = open(file_path, "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    predictions = []
    true_labels = []
    prob_val = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            crops, _, labels = batch 

            crops = crops.to(device)
            labels = labels.to(device)

            logits = model(crops)

            # import pdb; pdb.set_trace()
            probabilities = F.softmax(logits, dim=-1)
            pred = probabilities.max(1, keepdim=True)[1]
            greater_prob = [x[1] for x in probabilities.tolist()]
            
            predictions.extend([x[0] for x in pred.tolist()])
            true_labels.extend(labels.tolist())
            
            prob_val.extend(greater_prob)
            
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        print(f'Accuracy: {accuracy:.2f}')
        print(f'F1 Score: {f1:.2f}')
        file.write(f'Accuracy: {accuracy:.2f}\n')
        file.write(f'F1 Score: {f1:.2f}\n')

        print(classification_report(true_labels, predictions, labels=[0, 1]))
        file.write(classification_report(true_labels, predictions, labels=[0, 1]))

        auc_score = roc_auc_score(true_labels, prob_val)
        print('Logistic: ROC AUC=%.3f' % (auc_score))
        file.write('Logistic: ROC AUC=%.3f\n' % (auc_score))
        # calculate roc curves
        lr_fpr, lr_tpr, _ = roc_curve(true_labels, prob_val) 
        plt.plot(lr_fpr, lr_tpr, marker='.', label='text')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(plot_path)
        plt.show()

if __name__=="__main__":
    # TEST_CSV = "/home/cse/visitor/abansal.visitor/scratch/focalnet_dino/resnet50/data_csv/test_correct.csv"
    # TEST_IMG_BASE = "/home/cse/visitor/abansal.visitor/scratch/data/Test_Cropped"
    # TEST_TEXT_BASE = "/home/cse/visitor/abansal.visitor/scratch/focalnet_dino/cropped_data/Test_focalnet"
    TEST_CSV = "/home/cse/visitor/abansal.visitor/scratch/IRCH_DATA/irch_data.csv"
    TEST_IMG_BASE = "/home/cse/visitor/abansal.visitor/scratch/IRCH_DATA/Mammo_PNG"
    TEST_TEXT_BASE = "/home/cse/visitor/abansal.visitor/scratch/IRCH_DATA/Mammo_PNG_focalnet"

    checkpoint_path = "vision_table_1/vitb_16_clip/model_best.pt"
    num_workers = 8
    batch_size = 4
    plot_path = './vision_table_1/vitb_16_clip/result_auc_IRCH.png'
    score_file = './vision_table_1/vitb_16_clip/result_scores_IRCH.txt'
    topk = 8
    img_size = 224

    layers_freeze = 2

    print(f'topk = {topk}\nnum_workers = {num_workers}\nbatch_size = {batch_size}\nimage = {img_size}\nlayers_freeze = {layers_freeze}')

    model = load_model_again(checkpoint_path, layers_freeze, img_size)
    print("Loading validation DataLoader: ")
    val_dataloader = load_data(TEST_CSV, TEST_IMG_BASE, TEST_TEXT_BASE, num_workers, batch_size, topk, img_size)    
    print("Now Testing: ")
    test_code(model, val_dataloader, plot_path, score_file)

