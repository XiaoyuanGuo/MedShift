import os
import time
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torchvision.models as models

from dataset import Part_Emory_Allclass_Dataset
from torch.utils.data import DataLoader

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            
def initialize_model(num_classes, feature_extract, use_pretrained=True):
    model_ft= models.resnet152(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def eval_model(model, dataloader, criterion):

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    y_true = []
    y_pred = []
    y_prob = []
    # Iterate over data.
    for inputs,labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        y_true += list(labels.detach().cpu().numpy())
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred += list(preds.detach().cpu().numpy())
            y_prob += list(outputs.detach().cpu().numpy())
        running_corrects += torch.sum(preds == labels.data)
    acc = running_corrects.double() / len(dataloader.dataset)

    return  y_true, y_pred, y_prob  
  
def eval(criterion):
    checkpoint = torch.load("./weights/Resnet_152_internal.pth.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    ###start testing 
    stanford_label_dict = np.load("./label_dict.npy",allow_pickle=True).item()

    test_data_lists = []
    datasets = ["XR_HAND" ,"XR_FOREARM", "XR_SHOULDER" , "XR_FINGER" ,"XR_ELBOW" ,"XR_WRIST" ,"XR_HUMERUS"]
    for idx in range(0, len(datasets)):
        stanford_paths = stanford_label_dict[datasets[idx]]
        print(datasets[idx], len(stanford_paths))
        test_data = [[stanford_paths[i], idx] for i in range(0, len(stanford_paths))]
        test_data_lists += test_data
    test_mura_dataset = Part_Emory_Allclass_Dataset(test_data_lists, "test")
    test_mura_dataloader = DataLoader(test_mura_dataset, batch_size = batch_size, shuffle=False, num_workers = 1)
    y_true, y_pred, y_prob = eval_model_train(model, test_mura_dataloader, criterion)

    predictions = np.argmax(y_prob, axis=1)
    print(metrics.classification_report(y_true, predictions, digits=3))
    
    print("macro: ", roc_auc_score(np.array(y_true), nn.functional.softmax(torch.from_numpy(np.array(y_prob))), multi_class="ovr",average="macro"))
    print("weighted: ", roc_auc_score(np.array(y_true), nn.functional.softmax(torch.from_numpy(np.array(y_prob))), multi_class="ovr",average="weighted"))
    
    


def main():  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 7
    batch_size = 512
    num_epochs = 50
    start_epoch = 0
    feature_extract = True
    
    #collect all the emory data for classifier training
    # process dataset
    train_data_lists = []
    val_data_lists = []
    datasets = ["XR_HAND" ,"XR_FOREARM", "XR_SHOULDER" , "XR_FINGER" ,"XR_ELBOW" ,"XR_WRIST" ,"XR_HUMERUS"]
    for idx in range(0, len(datasets)):
        df = pd.read_csv("internal_train_"+datasets[idx]+".csv")
        train_data = [[df.iloc[i][1], idx] for i in range(0, int(0.8*len(df)))]
        train_data_lists += train_data
        val_data = [[df.iloc[i][1], idx] for i in range(int(0.8*len(df)), len(df))]
        val_data_lists += val_data

    train_mura_dataset = Part_Emory_Allclass_Dataset(train_data_lists, "train")
    train_mura_dataloader = DataLoader(train_mura_dataset, batch_size = batch_size, shuffle=True, num_workers = 1)
    val_mura_dataset = Part_Emory_Allclass_Dataset(val_data_lists, "val")
    val_mura_dataloader = DataLoader(val_mura_dataset, batch_size = batch_size, shuffle=False, num_workers = 1)

    dataloaders={"train":train_mura_dataloader, "val":val_mura_dataloader}

    model= initialize_model(num_classes, feature_extract, use_pretrained=True)
    model = model.to(device)

    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(params_to_update, lr=1e-3, weight_decay=1e-6, amsgrad=True)

    model.train()
    criterion = nn.CrossEntropyLoss()


    model, hist = train_model(model, dataloaders, criterion, optimizer, num_epochs)

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
               },"./weights/Resnet_152_emory.pth.tar")
    
     eval(criterion)
