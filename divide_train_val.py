import os
# set the visible devices to 7 here
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys


FOLDER_VAL = 'dataset_img/val'

try:
    os.makedirs(FOLDER_VAL)
except:
    pass

j = 0

for i in os.listdir('dataset_img/train/'):
    file = i

    # print(file)
    first_us_ind = file.find('_')
    # print(first_us_ind)
    second_us_ind = file.find('_', first_us_ind+1)
    # print(second_us_ind)
    dot_pos = file.find('.')
    # print(dot_pos)

    # print('Label:', file[second_us_ind+1 : dot_pos])
    # print('Index:', file[first_us_ind+1 : second_us_ind])
    label = int(file[second_us_ind+1 : dot_pos])
    idx = int(file[first_us_ind+1 : second_us_ind])


    if(idx>=40000):
        os.replace("dataset_img/train/train_" + str(idx)+ '_' + str(label) + '.png', "dataset_img/val/val_" + str(j) + '_' + str(label) + '.png')
        j+=1        
            