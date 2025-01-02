import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE

from model_vitb16_clip import CLIP_IMAGE
from data_da import all_mammo_da
    
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

def load_model_again(checkpoint_path, layers_freeze, img_size):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(device)

    model = CLIP_IMAGE(layers_freeze)
    model.load_state_dict(remove_module_prefix(torch.load(checkpoint_path)))
    model.to(device)

    model = torch.nn.DataParallel(model)

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


    
    plt.title('focal_crops_VITB16 CLIP INBreast t-SNE Plot of Embedding Values')
    # plt.title('focal_crops_VITDINO aiims t-SNE Plot of Embedding Values')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig(plot_save[:-7] + "tsne.png")
    plt.clf()

def test_code(model, test_dataloader, plot_path, file_path, logits_save):
    file = open(file_path, "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    predictions = []
    true_labels = []
    prob_val = []
    logits_labels = np.empty((0, 3)) 
    # embeddings = torch.empty((0, 384), dtype=torch.float32) #vitdino
    embeddings = torch.empty((0, 512), dtype=torch.float32)   #clip
    labels_tensor = torch.empty((0), dtype=torch.int)
    embeddings = embeddings.to(device)
    labels_tensor = labels_tensor.to(device)

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            crops, labels = batch 

            crops = crops.to(device)
            labels = labels.to(device)

            logits, features = model(crops)

            # import pdb; pdb.set_trace()
            probabilities = F.softmax(logits, dim=-1)
            pred = probabilities.max(1, keepdim=True)[1]
            greater_prob = [x[1] for x in probabilities.tolist()]
            
            
            
            embeddings = torch.cat((embeddings, features), 0)
            labels_tensor = torch.cat((labels_tensor, labels), 0)
            # logits_labels = np.vstack([logits_labels, greater_prob])
            
            predictions.extend([x[0] for x in pred.tolist()])
            true_labels.extend(labels.tolist())
            
            prob_val.extend(greater_prob)
            
            
        np.save(logits_save, logits_labels)
        np.save(logits_save[:-17] + "embeddings_save.npy", embeddings.cpu().numpy())
        tsne(embeddings, labels_tensor, plot_path)    
            
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
    
    TEST_CSV = "/home/tajamul/scratch/DA/Datasets/Foundation_Data/INBreast/test.csv"
    TEST_IMG_BASE = "/home/tajamul/scratch/DA/Datasets/Foundation_Data/INBreast/test2017"
    TEST_TEXT_BASE = "/home/tajamul/scratch/DA/Datasets/Foundation_Data/INBreast/test2017_focalnet_a25"

    checkpoint_path = "./models/vit16_clip/aiims/model_best.pt"
    num_workers = 8
    batch_size = 4
    plot_path = './models/vit16_clip/aiims/results_inbreast/result_auc_plot.png'
    score_file = './models/vit16_clip/aiims/results_inbreast/result_scores.txt'
    logits_save = './models/vit16_clip/aiims/results_inbreast/logits_labels.npy'
    topk = 8
    img_size = 224

    layers_freeze = 2

    print(f'topk = {topk}\nnum_workers = {num_workers}\nbatch_size = {batch_size}\nimage = {img_size}\nlayers_freeze = {layers_freeze}')

    model = load_model_again(checkpoint_path, layers_freeze, img_size)
    print("Loading Test DataLoader: ")
    val_dataloader = load_data(TEST_CSV, TEST_IMG_BASE, TEST_TEXT_BASE, num_workers, batch_size, topk, img_size)    
    print("Now Testing: ")
    test_code(model, val_dataloader, plot_path, score_file, logits_save)

