import numpy as np
import os
import pandas as pd
from utils import *

def preprocessing_tep(train_folder, test_folder, wavelet, time_step, level):

    preprocessed_data = {}

    ###################################################
    ####   '''loading tep_train dataset'''        #####
    ###################################################

    filenames = []
    train_feature = []
    train_lable = []

    for filename in os.listdir(train_folder):
        filenames.append(os.path.join(train_folder, filename))

    print(f"num of class, aka the amount of abnormality: {len(filenames)}")

    for i in range(len(filenames)):
        if filenames[i] == 'tep_train\\d00.dat': continue # we don't want normal data

        tep = np.genfromtxt(filenames[i])
        train_feature.append(tep)
        label = np.ones(tep.shape[0])*i
        train_lable.append(label)
        
    train_feature = np.array(train_feature)
    train_lable = np.array(train_lable)

    print(f'train_feature.shape: {train_feature.shape}')
    print(f'train_lable.shape: {train_lable.shape}')

    ###################################################
    ####         '''loading tep_test'''           #####
    ###################################################

    filenames_test = []
    test_normal = []
    test_fault = []
    val = []

    test_normal_label = []
    test_fault_label = []
    val_label = []

    for filename in os.listdir(test_folder):
        filenames_test.append(os.path.join(test_folder,filename))
        
    print("num of class, aka the amount of abnormality: ",len(filenames_test))

    for i in range(len(filenames_test)):
        if filenames[i] == 'tep_train\\d00.dat': continue # we don't want normal data
        tep = np.genfromtxt(filenames_test[i])

        # we also use normal data to validate our model.
        # theoretically, model will fail, because we only train the model on abnoraml data.
        test_normal_ = tep[:160] 
        test_normal.append(test_normal_)

        val_ = tep[160:160+300]
        val.append(val_)
        
        test_fault_ = tep[160+300:]
        test_fault.append(test_fault_)

        label = np.ones(tep.shape[0])*i
        test_normal_label.append(label[:160])
        test_fault_label.append(label[160+300:])
        val_label.append(label[160:160+300])
        
    test_normal = np.array(test_normal)
    test_fault = np.array(test_fault)
    val = np.array(val)
    test_normal_label = np.array(test_normal_label)
    test_fault_label = np.array(test_fault_label)
    val_label = np.array(val_label)

    print(f'test_normal.shape: {test_normal.shape}')
    print(f'test_normal_label.shape: {test_normal_label.shape}')
    print(f'val.shape: {val.shape}')
    print(f'val_label.shape: {val_label.shape}')
    print(f'test_fault.shape: {test_fault.shape}')
    print(f'test_fault_label.shape: {test_fault_label.shape}')

    if wavelet:
        print('*'*50)
        print(f'if use wavelet: {wavelet}')
        print(f'time_step: {time_step}')
        print(f'wavelet level: {level}')
        print('*'*50)

        #########################################################
        #### '''doing wavelet to inputting data(feature)''' #####
        #########################################################

        train_x = add_window_wavelet(train_feature, time_step, level)
        train_x = train_x.reshape([-1, tep.shape[-1]*((level+1)*2) ])
        train_y = label_add_window_wavelet(train_lable, time_step).reshape([-1])

        val_x = add_window_wavelet(val, time_step, level)
        val_x = val_x.reshape([-1, tep.shape[-1]*((level+1)*2) ])
        val_y = label_add_window_wavelet(val_label, time_step).reshape([-1])

        test_normal_x = add_window_wavelet(test_normal, time_step, level)
        test_normal_x = test_normal_x.reshape([-1, tep.shape[-1]*((level+1)*2) ])
        test_normal_y = label_add_window_wavelet(test_normal_label, time_step).reshape([-1])

        test_fault_x = add_window_wavelet(test_fault, time_step, level)
        test_fault_x = test_fault_x.reshape([-1, tep.shape[-1]*((level+1)*2) ])
        test_fault_y = label_add_window_wavelet(test_fault_label, time_step).reshape([-1])

        preprocessed_data["train_x"] = train_x
        preprocessed_data["train_y"] = train_y
        preprocessed_data["val_x"] = val_x
        preprocessed_data["val_y"] = val_y
        preprocessed_data["test_normal_x"] = test_normal_x
        preprocessed_data["test_normal_y"] = test_normal_y
        preprocessed_data["test_fault_x"] = test_fault_x
        preprocessed_data["test_fault_y"] = test_fault_y
    
    else:
        print('*'*50)
        print(f'if use wavelet: {wavelet}')
        print('*'*50)
        #############################################################
        #### '''NOT doing wavelet to inputting data(feature)''' #####
        #############################################################
        time_step = 1

        train_x = add_window(train_feature, time_step).reshape([-1, time_step*tep.shape[-1]])
        train_x, data_mean , data_std = normalization_1(train_x)
        train_y = label_add_window(train_lable, time_step).reshape([-1])

        val_x = add_window(val, time_step).reshape([-1, time_step*tep.shape[-1]])
        val_x = (val_x - data_mean)/data_std
        val_y = label_add_window(val_label, time_step).reshape([-1])

        test_normal_x = add_window(test_normal, time_step).reshape([-1, time_step*tep.shape[-1]])
        test_normal_x = (test_normal_x - data_mean)/data_std
        test_normal_y = label_add_window(test_normal_label, time_step).reshape([-1])

        test_fault_x = add_window(test_fault, time_step).reshape([-1, time_step*tep.shape[-1]])
        test_fault_x = (test_fault_x - data_mean)/data_std
        test_fault_y = label_add_window(test_fault_label, time_step).reshape([-1])

        preprocessed_data["train_x"] = train_x
        preprocessed_data["train_y"] = train_y
        preprocessed_data["val_x"] = val_x
        preprocessed_data["val_y"] = val_y
        preprocessed_data["test_normal_x"] = test_normal_x
        preprocessed_data["test_normal_y"] = test_normal_y
        preprocessed_data["test_fault_x"] = test_fault_x
        preprocessed_data["test_fault_y"] = test_fault_y

    num_class = len(filenames) # amount of class (how many abnormal types we train on the model)

    return preprocessed_data, num_class