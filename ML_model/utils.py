import numpy as np
import pywt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics

'''mean-std noralization'''
def normalization_1(data):
    
    data_mean = np.mean(data,0)   #(72,)
    data_std = np.std(data,0,ddof=1) 
    data_ = (data - data_mean)/data_std
    
    return data_, data_mean , data_std

'''max-min noralization'''
def normalization_2(data):
    
    data_min = np.min(data, 0)
    data_max = np.max(data, 0) 
    data_ = (data - data_min) / (data_max - data_min + 1e-7)
    
    return data_, data_min , data_max

'''one-hot label'''
# first arg: label
# second arg: num of class 
def one_hot ( labels , Label_class ): 
    one_hot_label = np.array([[ int (i == int (labels[j])) for i in range (Label_class)] for j in range ( len (labels))])      
    return one_hot_label

'''spilt them by slidding window (for no wavelet)'''
def add_window(x, time_step):
    
    x_window = []

    for i in range(x.shape[0]):
        series = x[i]
        series_window = []
        for j in range(int(series.shape[0]//time_step)):
            dat = series[j: j+time_step]
            series_window.append(dat.reshape([-1]))
        x_window.append(series_window)
    
    return np.array(x_window)

'''doing wavelet on input data, and spilt them by slidding window (for wavelet)'''
def add_window_wavelet(x, time_step, level_):
    
    x_window = []

    for i in range(x.shape[0]):
        series = x[i]
        series_window = []
        for j in range(int(series.shape[0]-time_step)):
            dat = series[j: j+time_step]
            fre_msg_var = []
            for k in range(dat.shape[-1]): #var_num

                coeffs = pywt.wavedec(dat[:,k], 'db1', level = level_, mode='sym')

                fre_msg_ = []
                for i in coeffs: # num_level * 3(mean + std + kurtosis)
                    fre_msg_.append(np.mean(i, 0))
                    fre_msg_.append(np.std(i, 0, ddof=1))
#                     fre_msg_.append(stats.kurtosis(i))
                fre_msg_var.append(fre_msg_)

            fre_msg_var = np.array(fre_msg_var)

            series_window.append(fre_msg_var.reshape([-1]))
        x_window.append(series_window)
        
    return np.array(x_window)


'''spilt labe by slidding window in order to be align with input data'''
def label_add_window(x, time_step):
    
    x_window = []

    for i in range(x.shape[0]):
        series = x[i]
        series_window = []
        for j in range(int(series.shape[0]//time_step)):
            dat = series[j: j+time_step]
            series_window.append(dat[0])
        x_window.append(series_window)
    
    return np.array(x_window)


'''spilt labe by slidding window in order to be align with input data'''
def label_add_window_wavelet(x, time_step):
    
    x_window = []

    for i in range(x.shape[0]):
        series = x[i]
        series_window = []
        for j in range(int(series.shape[0]-time_step)):
            dat = series[j: j+time_step]
            series_window.append(dat[0])
        x_window.append(series_window)
    
    return np.array(x_window)


def show_performance(label, preds):
    accuracy = accuracy_score(label, preds)

    acc = accuracy * 100.0
    recall = metrics.recall_score(label, preds, average='macro')
    f1 = metrics.f1_score(label, preds, average='macro')
    precesion = metrics.precision_score(label, preds, average='macro')
    res_confusion_matrix = confusion_matrix(label, preds)

    # performance['Accuracy'] = accuracy * 100.0
    # performance['Recall'] = metrics.recall_score(label, preds, average='macro')
    # performance['F1-score'] = metrics.f1_score(label, preds, average='macro')
    # performance['Precesion'] = metrics.precision_score(label, preds, average='macro')
    # performance['confusion_matrix'] = confusion_matrix(label, preds)

    return acc, recall, f1, precesion, res_confusion_matrix