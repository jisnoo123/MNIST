import os
# set the visible devices to 7 here
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn.metrics import confusion_matrix
# to append the path which leads to open cv for the current account
# sys.path.append('/home/jimut/anaconda3/lib/python3.9/site-packages')
import cv2
import argparse

###################################
# Import the necessary modules from the library
# sys.path.insert(0, '../../')
# from models import *
# from metrics import *
# from losses import *
###################################

# mostly torch imports and plot imports
import os
import sys
import cv2
# mostly torch imports and plot imports
import torch
import shutil
import glob
import pickle
import random
random.seed(42)
import colorama
from colorama import Fore, Style
import numpy as np
np.random.seed(42)
import torch.utils
import torchvision
from torch import optim
import torch.distributions
torch.manual_seed(42)
from tqdm import tqdm
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from matplotlib import rc, rcParams
from numpy import sin
from pathlib import Path
from sklearn.metrics import *
import umap

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")
print("Device to be used : ",device)

# import sys
# sys.path.insert(0, '../')
# from run_model import *


misclassified_folder = 'inference_folder/misclassified'

try:
    os.makedirs(misclassified_folder)
except:
    pass

y_actual = list()
y_pred = list()



# Use the data generator to load the dataset    
class DataGenerator(Dataset):
    
    def __init__(self, image_list):
        self.image_adr = image_list
    
    # number of images in the image list
    def __len__(self):
        return len(self.image_adr)

    # getting single pair of data
    def __getitem__(self, idx):
        image_name = self.image_adr[idx].split('/')[-1]

        first_us_ind = image_name.find('_')
        second_us_ind = image_name.find('_', first_us_ind+1)
        dot_pos = image_name.find('.')

        label = int(image_name[second_us_ind+1 : dot_pos])
        # label = torch.tensor(label)
        

        img = cv2.imread(self.image_adr[idx], cv2.IMREAD_UNCHANGED)

        # cv2.imwrite(f"img_{label}.png",img)
        # quit()

        return torch.FloatTensor(img), label, image_name


# The dataloader
def load_data(image_list, batch_size, num_workers, shuffle=True):
    dataset = DataGenerator(image_list)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return data_loader


# Get the image adresses in a list

def get_img_adr(folder, sub_folder):
    image_adr = list()

    for i in os.listdir(folder + '/' + sub_folder):
        image_adr.append(folder + '/' + sub_folder + '/' + i)
    
    return image_adr

# save checkpoint in pytorch
def save_ckp(checkpoint, checkpoint_path, save_after_epochs):
    if checkpoint['epoch'] % save_after_epochs == 0:
        torch.save(checkpoint, checkpoint_path)


# load checkpoint in pytorch
def load_ckp(checkpoint_path, model, model_opt):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model_opt.load_state_dict(checkpoint['optimizer'])
    return model, model_opt, checkpoint['epoch']

def test_epoch(test_loader, model, optimizer, epoch, hist_folder_name):
    model.eval()
    
    progress_bar = tqdm(enumerate(test_loader))
    total_loss = 0.0

    for step, (inp, label, file_name) in progress_bar:
        inp = inp.to(device)

        # print('label', label)
        y_actual.append(label)

        label = label.to(device)
        label = torch.tensor(label)

        # predicted label
        pred_label = model.forward(inp)

        #LOSS FUNCTIONS
        cross_entropy_loss = nn.CrossEntropyLoss()

        #CALCULATING LOSSES
        loss = cross_entropy_loss(pred_label, label)

        #LOSS TAKEN INTO CONSIDERATION
        total_loss += loss.item()
        
        pred_label = pred_label.cpu().numpy()

        y_pred.append(np.argmax(pred_label,axis=1)[0])
        
        label = label.detach().cpu().numpy()

        if label[0] != np.argmax(pred_label, axis=1)[0]:
            misclassified_label_folder = misclassified_folder + '/' + str(label[0])

            try:
                os.makedirs(misclassified_label_folder)
            except:
                pass
            
            
            #Misclassified
            inp_img = inp.cpu().numpy() #* 255
            # rotate the axis here
            inp_img = np.transpose(inp_img, (1, 2, 0))

            # folder = misclassified_folder + '/' + l
            file_name_save = misclassified_label_folder + '/' + str(step) + '_gt_' + str(label[0]) + '_pred_' + str(np.argmax(pred_label, axis=1)[0]) + '.png'
                          
            cv2.imwrite(file_name_save, inp_img)                                                                                            

        no_img_to_write = 40
        inf_folder_name = 'inference_folder'

        if(step < no_img_to_write):
            inp_img = inp.cpu().numpy() #* 255
            # rotate the axis here
            inp_img = np.transpose(inp_img, (1, 2, 0))
            
            #FOLDER PATH TO WRITE THE INFERENCES
            inference_folder = inf_folder_name + '/correct/' + str(label[0])
            
            try:
                os.makedirs(inference_folder)
            except:
                pass
            
            # save inference after every 50th epoch

            print("\n Saving inferences at epoch === ",epoch)
            # img_ = np.squeeze(np.squeeze(p_img[0]))
        

            # Basic inference during training
            file_name_save = inference_folder + '/' + str(step) + '_gt_' + str(label[0])+"_pred_" + str(np.argmax(pred_label,axis=1)[0])+".png"
            cv2.imwrite(file_name_save, inp_img) 
        
        for i in range(len(label)):
            with open("{}/roc_inf_test.txt".format(hist_folder_name), "a") as text_file:
                text_file.write("{} {}\n".format(int(label[0]), pred_label[0]))
            
        progress_bar.set_description("Epoch: {} -  Loss: {} ".format(epoch, total_loss))


    with open("{}/test_logs.txt".format(hist_folder_name), "a") as text_file:
        text_file.write("{} {}\n".format(epoch, total_loss))
    
    print(Fore.RED+"Test Epoch: {} |  Loss: {}".format(epoch, total_loss)+Style.RESET_ALL)
    print("---------------------------------------------------------------------------------------------------------------")


def test(test_loader, model, optimizer, n_epoch, resume, model_name, hist_folder_name, metrics_folder, umap_layer):

    #PATH TO SAVE THE CHECKPOINT
    checkpoint_path = "checkpoint/{}".format(model_name)

    epoch = n_epoch
    #IF TRAINING IS TO RESUMED FROM A CERTAIN CHECKPOINT
    if resume:
        model, optimizer, epoch = load_ckp(
            checkpoint_path, model, optimizer)

    with torch.no_grad():
        test_epoch(test_loader, model, optimizer, epoch, hist_folder_name)

    accuracy = accuracy_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred, average='micro')
    precision = precision_score(y_actual, y_pred, average = 'micro')

    print('Accuracy of Test is', accuracy)
    print('Recall of test is', recall)
    print('Precision of test is', precision)


    print('Plotting the confusion matrix')
    #Plot the confusion matrix
    plot_confusion_matrix(y_actual, y_pred, metrics_folder)
    print('Plotting Confusion Matrix done')


    print('Plotting UMAP')
    extract_features_and_plot_umap(test_loader, model, umap_layer, metrics_folder)
    print('Plotting UMAP completed')



def plot_confusion_matrix(y_actual, y_pred, metrics_folder):
    cm = confusion_matrix(y_actual, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted', fontsize = 18)
    plt.ylabel('Actual', fontsize = 18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    plt.savefig(metrics_folder + '/confusion_matrix.png')



def extract_features_and_plot_umap(test_loader, model, layer_name, metrics_folder):
    """Extract features from specified layer and create UMAP visualization"""
    
    features = []
    labels = []
    
    def hook_fn(module, input, output):
        features.append(output.detach().cpu().numpy())
    
    # Register hook on the specified layer
    layer = getattr(model, layer_name)
    hook = layer.register_forward_hook(hook_fn)
    
    model.eval()
    progress_bar = tqdm(enumerate(test_loader), desc="Extracting features")
    
    with torch.no_grad():
        for step, (inp, label, file_name) in progress_bar:
            inp = inp.to(device)
            
            # Forward pass through model
            _ = model.forward(inp)
            
            # Store labels
            if isinstance(label, list):
                labels.extend(label)
            else:
                labels.append(label.item() if torch.is_tensor(label) else label)
    
    # Clean up hook
    hook.remove()
    
    # Combine all features and labels
    features = np.vstack(features)
    labels = np.array(labels)
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', alpha=0.7, s=20)
    plt.colorbar(scatter, label='Class')
    plt.title(f'UMAP Visualization: {layer_name} Features', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=18)
    plt.ylabel('UMAP Dimension 2', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.savefig(metrics_folder + '/umap.png')
    plt.show()



class MNISTConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTConvNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Add channel dimension if missing and convert to float if needed
        if len(x.shape) == 3:  # [batch, height, width]
            x = x.unsqueeze(1)  # Add channel dimension -> [batch, 1, height, width]
        if x.dtype != torch.float32:
            x = x.float()
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # First FC block
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second FC block
        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer with softmax
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        
        return x



def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test', help='enter the test folder', required = True)
    parser.add_argument("--lr", help="Enter the learning rate.", required=True)
    parser.add_argument("--num_epochs", help="Enter the number of epochs.", required=True)
    parser.add_argument("--chkpt_name", help="Enter the checkpoint save name.", required=True)
    parser.add_argument("--history_folder_name", help="Enter the folder name for history dumps.", required=True)
    parser.add_argument('--metrics_folder', help='enter the metrics folder', required = True)
    parser.add_argument('--umap_layer', help='enter the umap layer', required = True)

    test_image_address_list = get_img_adr("../dataset_img", "{}".format(parser.parse_args().test))
    
    # print('Test image addr list:', test_image_address_list)
    # #CREATING THE TEST LOADER
    test_loader = load_data(
        test_image_address_list, batch_size=1, num_workers=8, shuffle=False)

    # print('Test loader', test_loader)
    
    ############################
    
    #CALLING THE MODEL
    model = MNISTConvNet(num_classes=10)
    model = model.to(device)

    #DEFINING THE OPTIMIZER
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=float(parser.parse_args().lr), weight_decay=5e-4)
    
    n_epoch = int(parser.parse_args().num_epochs)

    #INDICATOR VARIABLE TO RESUME TRAINING OR START AFRESH
    resume = True
    
    hist_folder_name = parser.parse_args().history_folder_name

    model_name = parser.parse_args().chkpt_name

    metrics_folder = parser.parse_args().metrics_folder
    umap_layer = parser.parse_args().umap_layer
    
    try:
        os.makedirs(metrics_folder)
    except:
        pass

    test(test_loader, model, optimizer, n_epoch, resume, model_name, hist_folder_name, metrics_folder, umap_layer)



if __name__ == "__main__":
    print("--- Starting the main function ----")
    main()
