import os
# set the visible devices to 7 here
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys

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
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")
print("Device to be used : ",device)


FOLDER_NAME = 'checkpoint'
# Create the checkpoint folder if it does not exist
try:
    os.makedirs('checkpoint')
except:
    pass


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


# train epoch
def train_epoch(train_loader, model, optimizer, epoch, hist_folder_name):
    print("\n\n---------------------------------------------------------------------------------------------------------------\n")

    # print("train_loader = ",train_loader.)
    # quit()
    progress_bar = tqdm(enumerate(train_loader))
    total_loss = 0.0
    
    y_actual = list()
    y_pred = list()
    scores = list()

    N = 0
    for step, (inp, label, file_name) in progress_bar:
        
        model.train()

        for i in range(len(label)):
            y_actual.append(int(label[i]))
       

        label = torch.tensor(label)
        
        #TRANSFERRING DATA TO DEVICE
        inp = inp.to(device)
        label = label.to(device)

        # print("inp = ",inp," label = ",label)
        # quit()

        # clear the gradient
        optimizer.zero_grad()

        #GETTING THE PREDICTED LABEL
        pred_label = model.forward(inp)
        
        
        #LOSS FUNCTIONS
        cross_entropy_loss = nn.CrossEntropyLoss()

        # print("Label = ",label, "pred_label = ",pred_label)
        # print("Label = ",label.shape, "pred_label = ",pred_label.shape)
        # quit()

        #CALCULATING LOSSES
        loss = cross_entropy_loss(pred_label, label)

        pred_label = pred_label.detach().cpu().numpy()

        # print('pred_label', np.argmax(pred_label, axis=1))

        # print('pred_label', pred_label)

        label = label.detach().cpu().numpy()

        for i in range(len(pred_label)):
            scores.append(pred_label[i][label[i]])
        
        

        for i in range(len(np.argmax(pred_label, axis=1))):
            y_pred.append(int(np.argmax(pred_label, axis=1)[i]))

        #LOSS TAKEN INTO CONSIDERATION
        total_loss += loss.item()

        # print(loss)

        #BACKPROPAGATING THE LOSS
        loss.backward()
        optimizer.step()


        #DISPLAYING THE LOSS
        progress_bar.set_description("Epoch: {} -  Loss: {} ".format(epoch, loss))


    # Compute the precision
    precision = precision_score(y_actual, y_pred, average='macro')
    print('Precision is ', precision)

    # Compute the recall
    recall = recall_score(y_actual, y_pred, average = 'macro')
    print('Recall is ', recall)

    # Compute the accuracy
    accuracy = accuracy_score(y_actual, y_pred)
    print('Accuracy is:', accuracy)
    


    # Write out the accuracy, precision and recall in seperate files
    with open("{}/prec_train.txt".format(hist_folder_name), "a") as text_file:
        text_file.write("{} {}\n".format(epoch, precision))

    with open("{}/acc_train.txt".format(hist_folder_name), "a") as text_file:
        text_file.write("{} {}\n".format(epoch, accuracy))

    with open("{}/rec_train.txt".format(hist_folder_name), "a") as text_file:
        text_file.write("{} {}\n".format(epoch, recall))

    # Write the true label and scores in a file
    with open("{}/roc_inf_train.txt".format(hist_folder_name), "a") as text_file:
        text_file.write("{} {}\n".format(epoch, y_actual, scores))

    with open("{}/train_logs.txt".format(hist_folder_name), "a") as text_file:
        text_file.write("{} {}\n".format(epoch, total_loss))

    # print("Training Epoch: {} |  Total Loss: {} | Total Dice: {} | Total Jaccard: {} | N: {}".format(epoch,total_loss, total_dice, total_jacard,N))
    print(Fore.GREEN+"Training Epoch: {} |  Loss: {}".format(epoch, total_loss)+Style.RESET_ALL)

    return model, optimizer


def val_epoch(val_loader, model, optimizer, epoch, hist_folder_name):
    print("\n\n---------------------------------------------------------------------------------------------------------------\n")

    progress_bar = tqdm(enumerate(val_loader))
    total_loss = 0.0
    y_actual = list()
    y_pred = list()
    scores = list()

    N = 0
    for step, (inp, label, file_name) in progress_bar:
        
        model.eval()
        
        #TRANSFERRING DATA TO DEVICE
        inp = inp.to(device)

        for i in range(len(label)):
            y_actual.append(int(label[i]))

        label = label.to(device)
        label = torch.tensor(label)

        # clear the gradient
        # optimizer.zero_grad()

        # getting the predicted label
        pred_label = model.forward(inp)

        #LOSS FUNCTIONS
        cross_entropy_loss = nn.CrossEntropyLoss()

        #CALCULATING LOSSES
        loss = cross_entropy_loss(pred_label, label)

        #LOSS TAKEN INTO CONSIDERATION
        total_loss += loss.item()

        label = label.detach().cpu().numpy()

        for i in range(len(pred_label)):
            scores.append(pred_label[i][label[i]])


        pred_label = pred_label.detach().cpu().numpy()
        for i in range(len(np.argmax(pred_label, axis=1))):
            y_pred.append(int(np.argmax(pred_label, axis=1)[i]))
        # print(loss)

        #BACKPROPAGATING THE LOSS
        # loss.backward()
        # optimizer.step()

        #DISPLAYING THE LOSS
        progress_bar.set_description("Epoch: {} -  Loss: {} ".format(epoch, loss))

    # Compute the precision
    
    
    precision = precision_score(y_actual, y_pred, average='macro')
    print('Precision is ', precision)

    # Compute the recall
    recall = recall_score(y_actual, y_pred, average = 'macro')
    print('Recall is ', recall)

    # Compute the accuracy
    accuracy = accuracy_score(y_actual, y_pred)
    print('Accuracy is:', accuracy)
    
    # Write out the accuracy, precision and recall in seperate files
    with open("{}/prec_val.txt".format(hist_folder_name), "a") as text_file:
        text_file.write("{} {}\n".format(epoch, precision))

    with open("{}/acc_val.txt".format(hist_folder_name), "a") as text_file:
        text_file.write("{} {}\n".format(epoch, accuracy))

    with open("{}/rec_val.txt".format(hist_folder_name), "a") as text_file:
        text_file.write("{} {}\n".format(epoch, recall))

    # Write the true label and scores in a file
    with open("{}/roc_inf_val.txt".format(hist_folder_name), "a") as text_file:
        text_file.write("{} {}\n".format(epoch, y_actual, scores))

    with open("{}/val_logs.txt".format(hist_folder_name), "a") as text_file:
        text_file.write("{} {}\n".format(epoch, total_loss))

    print(Fore.GREEN+"Validation Epoch: {} |  Loss: {}".format(epoch, total_loss)+Style.RESET_ALL)


def train_val(train_loader, val_loader, model, optimizer, n_epoch, resume, model_name, hist_folder_name, save_after_epochs):

    Path(hist_folder_name).mkdir(parents=True, exist_ok=True)

    epoch = 0

    #PATH TO SAVE THE CHECKPOINT
    checkpoint_path = "checkpoint/{}_{}.pt".format(model_name,epoch)

    #IF TRAINING IS TO RESUMED FROM A CERTAIN CHECKPOINT
    if resume:
        model, optimizer, epoch = load_ckp(
            checkpoint_path, model, optimizer)

    while epoch <= n_epoch:
        checkpoint_path = "checkpoint/{}_{}.pt".format(model_name,epoch)
        epoch += 1
        model, optimizer = train_epoch(train_loader, model, optimizer, epoch, hist_folder_name)
        
        #CHECKPOINT CREATION
        checkpoint = {'epoch': epoch+1, 'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        
        #CHECKPOINT SAVING
        save_ckp(checkpoint, checkpoint_path, save_after_epochs)
        print(Fore.BLACK+"Checkpoint Saved"+Style.RESET_ALL)

        with torch.no_grad():
            val_epoch(val_loader, model, optimizer, epoch, hist_folder_name)



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
    
    parser.add_argument("--lr", help="Enter the learning rate.", required=True)
    parser.add_argument("--batch_size", help="Enter the batch size.", required=True)
    parser.add_argument("--num_epochs", help="Enter the number of epochs.", required=True)

    parser.add_argument("--train_images", help="Enter the folder name for train images.", required=True)
    parser.add_argument("--val_images", help="Enter the folder name for validation images.", required=True)
    parser.add_argument("--test_images", help="Enter the folder name for test images.", required=True)


    parser.add_argument("--history_folder_name", help="Enter the folder name for history dumps.", required=True)
    parser.add_argument("--chkpt_name", help="Enter the checkpoint save name.", required=True)
    parser.add_argument("--save_after_epoch", help="Enter the number of epochs after which saving needs to be done.", required=True)
    
    train_image_address_list = get_img_adr("../dataset_img/", "{}".format(parser.parse_args().train_images))

    print("Total Number of Training Images : ", len(train_image_address_list))

    val_image_address_list = get_img_adr("../dataset_img/", "{}".format(parser.parse_args().val_images))

    test_image_address_list = get_img_adr("../dataset_img/", "{}".format(parser.parse_args().test_images))


    save_after_epochs = int(parser.parse_args().save_after_epoch)

    # CREATING THE TRAIN LOADER
    train_loader = load_data(
        train_image_address_list, batch_size=int(parser.parse_args().batch_size), num_workers=8, shuffle=True)

    val_loader = load_data(
        val_image_address_list,  batch_size=int(parser.parse_args().batch_size), num_workers=8, shuffle=True)
    

    # Initialize model
    model = MNISTConvNet(num_classes=10)
    #CALLING THE MODEL
    # model = NeuralNetwork()
    # model = nn.DataParallel(model)
    model = model.to(device)

    # summary(model, input_size=(3, 320, 192))

    #DEFINING THE OPTIMIZER
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=float(parser.parse_args().lr), weight_decay=5e-4)
    
    n_epoch = int(parser.parse_args().num_epochs)

    #INDICATOR VARIABLE TO RESUME TRAINING OR START AFRESH
    resume = False
    model_name = parser.parse_args().chkpt_name
    hist_folder_name = parser.parse_args().history_folder_name
    
    train_val(train_loader, val_loader, model, optimizer, n_epoch, resume, model_name, hist_folder_name, save_after_epochs)



if __name__ == "__main__":
    print("--- Starting the main function ----")
    main()
