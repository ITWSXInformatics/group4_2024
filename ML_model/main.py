import xgboost as xgb
import numpy as np
import click
import pandas as pd
from utils import *
from preprocessing import preprocessing_tep 

@click.command()
@click.argument("train_folder", default="tep_train")
@click.argument("test_folder", default="tep_test")
@click.argument("time_step", default = 110) # "the size of slidding window"
@click.argument("level", default = 7) # "the level of wavelet"
@click.option('--wavelet', is_flag=True, help='if using wavelet')

def train(
    train_folder,
    test_folder,
    wavelet: bool,
    time_step,
    level,
):
    preprocessed_tep, num_class = preprocessing_tep(train_folder, test_folder, wavelet, time_step, level)
    print(preprocessed_tep.keys())

    train_x = preprocessed_tep['train_x']
    train_y = preprocessed_tep['train_y']
    val_x = preprocessed_tep['val_x']
    val_y = preprocessed_tep['val_y']
    test_normal_x = preprocessed_tep['test_normal_x']
    test_normal_y = preprocessed_tep['test_normal_y']
    test_fault_x = preprocessed_tep['test_fault_x']
    test_fault_y = preprocessed_tep['test_fault_y']

    print(f'train_x : {train_x.shape}')
    print(f'train_y : {train_y.shape}')
    print(f'val_x : {val_x.shape}')
    print(f'val_y : {val_y.shape}')
    print(f'test_normal_x : {test_normal_x.shape}')
    print(f'test_normal_y : {test_normal_y.shape}')
    print(f'test_fault_x : {test_fault_x.shape}')
    print(f'test_fault_y : {test_fault_y.shape}')

    dtrain = xgb.DMatrix(train_x, train_y)
    dval = xgb.DMatrix(val_x, val_y)
    dtest_normal = xgb.DMatrix(test_normal_x, test_normal_y)
    dtest_fault = xgb.DMatrix(test_fault_x, test_fault_y)

    ############################################
    ####       '''training model '''       #####
    ############################################

    params = {
    'objective': 'multi:softprob',
    'num_class': num_class,
    'seed': 0,
    'gamma': 0.5,
    'max_depth': 2, #10
    'random_state': 0,
    'subsample': 1,
    'min_child_weight': 3,
    'lambda': 3,
    'grow_policy': 'lossguide',
    'eta': 0.007,
    'eval_metric': ['merror'],
    }

    model = xgb.train(params, dtrain, 
            num_boost_round = 4000,  # 4000
            verbose_eval = 20, 
            early_stopping_rounds = 200, 
            evals=[(dtrain, 'train') , (dval, 'valid'), (dtest_normal, 'test_normal'), (dtest_fault, 'test_fault')],
            )

    print('training finish')

    ############################################################
    ####     '''show validation dataset performance '''    #####
    ############################################################

    y_pred = model.predict(xgb.DMatrix(val_x))
    yprob = np.argmax(y_pred, axis=1)  # return the index of the biggest pro
    predictions = [round(value) for value in yprob]

    acc, recall, f1, precesion, confusion_matrix = show_performance(val_y, predictions)

    print()
    print('*'*50)
    print('Val performance: ')
    print("Accuracy: %.5f%%" % acc)
    print('Recall: %.4f' % recall)
    print('F1-score: %.4f' % f1)
    print('Precesion: %.4f' % precesion)
    print("confusion_matrix:")
    print(confusion_matrix)
    print('*'*50)
    print()

    # save performance to csv
    df_val_perform = pd.DataFrame()
    df_val_perform['Accuracy'] = [acc]
    df_val_perform['Recall'] = [recall]
    df_val_perform['F1-score'] = [f1]
    df_val_perform['Precesion'] = [precesion]
    if wavelet:
        df_val_perform.to_csv(f"val_perform_wavelet.csv", index=False)
    else:
        df_val_perform.to_csv(f"val_perform.csv", index=False)

    ################################################################################################
    ####     '''show normal dataset performance '''                                            #####
    ####     theoretically, model will fail, because we only train the model on abnoraml data. #####
    ################################################################################################

    y_pred = model.predict(xgb.DMatrix(test_normal_x))
    yprob = np.argmax(y_pred, axis=1)  # return the index of the biggest pro
    predictions = [round(value) for value in yprob]

    acc, recall, f1, precesion, confusion_matrix = show_performance(test_normal_y, predictions)

    print()
    print('*'*50)
    print('test_noraml performance: ')
    print("Accuracy: %.5f%%" % acc)
    print('Recall: %.4f' % recall)
    print('F1-score: %.4f' % f1)
    print('Precesion: %.4f' % precesion)
    print("confusion_matrix:")
    print(confusion_matrix)
    print('*'*50)
    print()

    # save performance to csv
    df_test_noraml_perform = pd.DataFrame()
    df_test_noraml_perform['Accuracy'] = [acc]
    df_test_noraml_perform['Recall'] = [recall]
    df_test_noraml_perform['F1-score'] = [f1]
    df_test_noraml_perform['Precesion'] = [precesion]
    if wavelet:
        df_test_noraml_perform.to_csv(f"test_noraml_perform_wavelet.csv", index=False)
    else:
        df_test_noraml_perform.to_csv(f"test_noraml_perform.csv", index=False)


    ################################################################
    ####     '''show abnormal testing dataset performance '''  #####
    ################################################################

    y_pred = model.predict(xgb.DMatrix(test_fault_x))
    yprob = np.argmax(y_pred, axis=1)  # return the index of the biggest pro
    predictions = [round(value) for value in yprob]

    acc, recall, f1, precesion, confusion_matrix = show_performance(test_fault_y, predictions)

    print()
    print('*'*50)
    print('test_abnoraml performance: ')
    print("Accuracy: %.5f%%" % acc)
    print('Recall: %.4f' % recall)
    print('F1-score: %.4f' % f1)
    print('Precesion: %.4f' % precesion)
    print("confusion_matrix:")
    print(confusion_matrix)
    print('*'*50)
    print()

    # save performance to csv
    df_test_abnoraml_perform = pd.DataFrame()
    df_test_abnoraml_perform['Accuracy'] = [acc]
    df_test_abnoraml_perform['Recall'] = [recall]
    df_test_abnoraml_perform['F1-score'] = [f1]
    df_test_abnoraml_perform['Precesion'] = [precesion]
    if wavelet:
        df_test_abnoraml_perform.to_csv(f"test_abnoraml_perform_wavelet.csv", index=False)
    else:
        df_test_abnoraml_perform.to_csv(f"test_abnoraml_perform.csv", index=False)

    print(f"df_test_abnoraml_perform: {df_test_abnoraml_perform}")


if __name__ == "__main__":
    train()