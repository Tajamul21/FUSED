# Importing the Libraries:
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import numpy as np

from args import get_args
from model_vitb16_dino import vitdino
from data import all_mammo
from data_da import all_mammo_da
from test import test_code, load_model_again

# from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.sampler import WeightedRandomSampler

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

seed_value = 42
np.random.seed(seed_value)

def make_weights(train_targets, prob_malignant):
    class_sample_counts = [train_targets.count(class_idx) for class_idx in range(2)]
    weights = np.array(class_sample_counts, dtype=np.float32)

    class_weights = [1-prob_malignant, prob_malignant] / weights
    # import pdb; pdb.set_trace()
    train_targets = np.array(train_targets, dtype=np.float64)
    train_targets[np.where(np.array(train_targets)==0)[0]] = class_weights[0]
    train_targets[np.where(np.array(train_targets)==1)[0]] = class_weights[1]
    # Define sampler for weighted sampling on the training set
    train_targets = torch.tensor(train_targets) 

    return train_targets

def load_data(CSV, IMG_BASE, TEXT_BASE, prob_malignant=0.5, type=1, workers=8, batch_size=32, topk=5, img_size=224):
    # 1 for train 0 for test

    dataset = all_mammo_da(CSV, IMG_BASE, TEXT_BASE, topk=topk, img_size=img_size)
    # Calculate class weights for weighted sampling on the training set
    print(f'Malignancy Count: {(sum(dataset.label) / len(dataset.label)) * 100 if dataset.label else 0}')
    train_targets = dataset.label
    train_targets = make_weights(train_targets, prob_malignant)
    train_targets = train_targets.to("cuda")
     
    if type == 1:
        sampler = WeightedRandomSampler(train_targets, train_targets.shape[0]*2, replacement=True)
        print("Made Train Dataloader")
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,drop_last=True) 
    else: 
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, drop_last=True) 
        print("Made Test Dataloader")

    return dataloader

def load_model(layers_freeze, img_size):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(device)

    model = vitdino(layers_freeze, img_size)
    model.to(device)

    model = torch.nn.DataParallel(model)

    return model

def train_code(model, train_dataloader, val_dataloader, file_path, checkpoint_path, plot_path, num_epochs=50, learning_rate=5e-3):
    file = open(file_path, "w")
    file.close()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9,0.98),eps=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 if epoch == 19 else 1)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_max = sys.maxsize
    loss_list_val = []
    loss_list_train = []
    exit_cnt = 0

    for epoch in range(num_epochs):
        file = open(file_path, "a")
        print(f'Started Epoch #{epoch+1}')

        pbar_train = tqdm(train_dataloader, total=len(train_dataloader), desc='train', position=0, leave=True)

        model.train()
        avg_loss_train = 0
        batch_num_train = 0
        for batch in pbar_train:
            optimizer.zero_grad()
            crops, labels = batch 

            crops = crops.to(device)
            labels = labels.to(device)

            logits, _ = model(crops)
            loss = loss_criterion(logits, labels)

            avg_loss_train += loss.item()
            batch_num_train += 1

            loss.backward()
            optimizer.step()

            pbar_train.set_description(f"\tEpoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        avg_loss_train /= batch_num_train 
        tqdm.write(f'Epoch {epoch+1}: Average loss TRAIN = {avg_loss_train:.4f}')
        file.write(f'Epoch {epoch+1}: Average loss TRAIN = {avg_loss_train:.4f}\n')

        # Validation
        model.eval()
        pbar_test = tqdm(val_dataloader, total=len(val_dataloader), desc='val', position=0, leave=True)
        avg_loss_val = 0
        batch_num_val = 0
        with torch.no_grad():
            for batch in pbar_test:
                crops, labels = batch 

                crops = crops.to(device)
                labels = labels.to(device)

                logits, _ = model(crops)
                loss = loss_criterion(logits, labels)

                avg_loss_val += loss.item()
                batch_num_val += 1
                pbar_test.set_description(f"\tEpoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
            avg_loss_val /= batch_num_val

        tqdm.write(f'Epoch {epoch+1}: Average loss VAL = {avg_loss_val:.4f}')
        file.write(f'Epoch {epoch+1}: Average loss VAL = {avg_loss_val:.4f}\n')
        # scheduler.step()
        print(f'Epoch {epoch + 1}: Learning Rate: {optimizer.param_groups[0]["lr"]}')

        loss_list_val.append(avg_loss_val)
        loss_list_train.append(avg_loss_train) 
        plt.plot([num+1 for num in range(len(loss_list_val))], loss_list_val, label = "VAL_LOSS")
        plt.plot([num+1 for num in range(len(loss_list_val))], loss_list_train, label = "TRAIN_LOSS")
        plt.xlabel('Epoch #')
        plt.ylabel('Validation & Train Loss')
        plt.legend()
        plt.savefig(plot_path)
        plt.clf()
        tqdm.write('\n\n')

        # scheduler.step(avg_loss_val)

        if avg_loss_val < val_max:
            val_max = avg_loss_val
            exit_cnt = 0
            torch.save(model.state_dict(), checkpoint_path)

            tqdm.write(f'\tEpoch #{epoch+1} - Model checkpoint saved.')
            file.write(f'\tEpoch #{epoch+1} - Model checkpoint saved.\n')
        else: 
            exit_cnt += 1
            if exit_cnt >= 30:
                tqdm.write(f'\Exiting training loop due to overfitting.')
                file.write(f'\Exiting training loop due to overfitting.\n')
                break

        file.close()
    
    best_lr_used = optimizer.param_groups[0]['lr']
    print(f'Best Learning Rate Used: {best_lr_used}')


if __name__=="__main__":
    TRAIN_CSV = "/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/Natural/cityscapes/cityscapes_source/train_id.csv"
    TRAIN_IMG_BASE = "/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/Natural/cityscapes/cityscapes_source/train2017"
    TRAIN_TEXT_BASE = "/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/Natural/cityscapes/cityscapes_source/train2017_focalnet79_crops"

    TEST_CSV = "/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/Natural/cityscapes/cityscapes_source/test_id.csv"
    TEST_IMG_BASE = "/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/Natural/cityscapes/cityscapes_source/test2017"
    TEST_TEXT_BASE = "/home/kartik_anand/scratch/kstyles/miccai/NEGRONI_DATASETS/Natural/cityscapes/cityscapes_source/test2017_focalnet79_crops"

    args = get_args()

    checkpoint_path = os.path.join(args.checkpoint_model_save + "model_best.pt")
    file_path = os.path.join(args.checkpoint_model_save + "training_stats.txt")
    plot_path = os.path.join(args.checkpoint_model_save + "loss_plot.png")
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    # DataLoader Params
    topk = args.topk
    num_workers = args.num_workers
    batch_size = args.batch_size
    prob_malignant = 0.3

    # Model Params
    layers_freeze = args.layers_freeze
    img_size = args.img_size

    print(f'topk={topk}\nnum_workers={num_workers}\nbatch_size={batch_size}\nlayers_freeze={layers_freeze}\nimagesize={img_size}')

    # import pdb; pdb.set_trace()

    model = load_model(layers_freeze, img_size)
    print("Loading training DataLoader: ")
    train_dataloader = load_data(TRAIN_CSV, TRAIN_IMG_BASE, TRAIN_TEXT_BASE, prob_malignant, 1, num_workers, batch_size, topk, img_size)  
    print("Loading validation DataLoader: ")
    val_dataloader = load_data(TEST_CSV, TEST_IMG_BASE, TEST_TEXT_BASE, prob_malignant, 0, num_workers, batch_size, topk, img_size)    
    print("Now training: \n\n")
    train_code(model, train_dataloader, val_dataloader, file_path, checkpoint_path, plot_path, num_epochs=num_epochs, learning_rate=learning_rate)

    checkpoint_path_test = os.path.join(args.checkpoint_model_save + "model_best.pt")
    plot_path_test = os.path.join(args.checkpoint_model_save + "result_auc_plot.png")
    score_file = os.path.join(args.checkpoint_model_save + "result_scores.txt")

    # test_model = load_model_again(checkpoint_path_test, layers_freeze, img_size)
    # test_code(test_model, val_dataloader, plot_path_test, score_file)


