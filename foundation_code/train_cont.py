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
from resnet50.code.model_rn50_imgnet import resnet
from data import all_mammo
from test import test_code, load_model_again
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import WeightedRandomSampler

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

seed_value = 42
np.random.seed(seed_value)

def load_data(CSV, IMG_BASE, TEXT_BASE, type=1, workers=8, batch_size=32, topk=5, img_size=224):
    # 1 for train 0 for test

    dataset = all_mammo(CSV, IMG_BASE, TEXT_BASE, topk=topk, img_size=img_size)
    # Calculate class weights for weighted sampling on the training set
    print(f'Malignancy Count: {(sum(dataset.label) / len(dataset.label)) * 100 if dataset.label else 0}')
    train_targets = dataset.label
    class_sample_counts = [train_targets.count(class_idx) for class_idx in range(2)]
    
    class_weights = 1.0 / np.array(class_sample_counts, dtype=np.float32)
    train_targets = np.array(train_targets, dtype=np.float64)
    train_targets[np.where(np.array(train_targets)==0)[0]] = class_weights[0]
    train_targets[np.where(np.array(train_targets)==1)[0]] = class_weights[1]
    # Define sampler for weighted sampling on the training set
    train_targets = torch.tensor(train_targets / np.sum(train_targets))  # Normalize to sum to 1
    
    train_targets = train_targets.to("cuda")
     
    if type == 1:
        sampler = WeightedRandomSampler(train_targets, train_targets.shape[0]*2, replacement=True)
        print("Made Train Dataloader")
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers) 
    else: 
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers) 
        print("Made Test Dataloader")

    return dataloader

def load_model(layers_freeze, img_size):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(device)

    model = resnet(layers_freeze, img_size)
    model.to(device)

    model = torch.nn.DataParallel(model)

    return model

def train_code(model, train_dataloader, val_dataloader, file_path, checkpoint_path, plot_path, num_epochs=50, learning_rate=5e-3):
    file = open(file_path, "w")
    file.close()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9,0.98),eps=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
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
            crops, _, labels = batch 

            crops = crops.to(device)
            labels = labels.to(device)

            logits = model(crops)
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
                crops, _, labels = batch 

                crops = crops.to(device)
                labels = labels.to(device)

                logits = model(crops)

                # Forward pass
                loss = loss_criterion(logits, labels)

                avg_loss_val += loss.item()
                batch_num_val += 1
                pbar_test.set_description(f"\tEpoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
            avg_loss_val /= batch_num_val

        tqdm.write(f'Epoch {epoch+1}: Average loss VAL = {avg_loss_val:.4f}')
        file.write(f'Epoch {epoch+1}: Average loss VAL = {avg_loss_val:.4f}\n')

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
            if exit_cnt >= 50:
                tqdm.write(f'\Exiting training loop due to overfitting.')
                file.write(f'\Exiting training loop due to overfitting.\n')
                break

        file.close()
    
    best_lr_used = optimizer.param_groups[0]['lr']
    print(f'Best Learning Rate Used: {best_lr_used}')


if __name__=="__main__":
    TRAIN_CSV = "/home/cse/visitor/abansal.visitor/scratch/focalnet_dino/resnet50/data_csv/train_final.csv"
    TRAIN_IMG_BASE = "/home/cse/visitor/abansal.visitor/scratch/data/Train_Cropped"
    TRAIN_TEXT_BASE = "/home/cse/visitor/abansal.visitor/scratch/focalnet_dino/cropped_data/Train_focalnet"

    TEST_CSV = "/home/cse/visitor/abansal.visitor/scratch/focalnet_dino/resnet50/data_csv/test_final.csv"
    TEST_IMG_BASE = "/home/cse/visitor/abansal.visitor/scratch/data/Test_Cropped"
    TEST_TEXT_BASE = "/home/cse/visitor/abansal.visitor/scratch/focalnet_dino/cropped_data/Test_focalnet"

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

    # Model Params
    layers_freeze = args.layers_freeze
    img_size = args.img_size

    print(f'topk={topk}\nnum_workers={num_workers}\nbatch_size={batch_size}\nlayers_freeze={layers_freeze}\nimagesize={img_size}')

    # import pdb; pdb.set_trace()

    checkpoint_path_train = "/home/cse/visitor/abansal.visitor/scratch/focalnet_dino/resnet50/models/r50_2_8_t2_1/model_best.pt"
    model = load_model_again(checkpoint_path_train, layers_freeze, img_size)
    print("Loading training DataLoader: ")
    train_dataloader = load_data(TRAIN_CSV, TRAIN_IMG_BASE, TRAIN_TEXT_BASE, 1, num_workers, batch_size, topk, img_size)  
    print("Loading validation DataLoader: ")
    val_dataloader = load_data(TEST_CSV, TEST_IMG_BASE, TEST_TEXT_BASE, 0, num_workers, batch_size, topk, img_size)    
    print("Now training: \n\n")
    train_code(model, train_dataloader, val_dataloader, file_path, checkpoint_path, plot_path, num_epochs=num_epochs, learning_rate=learning_rate)

    checkpoint_path_test = os.path.join(args.checkpoint_model_save + "model_best.pt")
    plot_path_test = os.path.join(args.checkpoint_model_save + "result_auc_plot.png")
    score_file = os.path.join(args.checkpoint_model_save + "result_scores.txt")

    test_model = load_model_again(checkpoint_path_test, layers_freeze, img_size)
    test_code(test_model, val_dataloader, plot_path_test, score_file)


    

