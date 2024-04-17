import xgboost as xgb
import numpy as np
import click
import pandas as pd
from utils import *

@click.command()
@click.argument("inputs", default="fault7_input.csv")
@click.argument("labels", default="fault7_output.csv")


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

def testing(
    inputs,
    labels,
):
    ####     '''load data(variables and labels) '''  #####
    test_fault_x = pd.read_csv(inputs)
    test_fault_y = pd.read_csv(labels)
    print(f"test_fault_x: {test_fault_x.shape}")
    print(f"test_fault_y: {test_fault_y.shape}")

    ####     '''load pretrainde model '''  #####
    model = xgb.Booster(model_file='xgb_model.bin') 

    ####     '''show abnormal testing dataset performance '''  #####
    y_pred = model.predict(xgb.DMatrix(test_fault_x))
    yprob = np.argmax(y_pred, axis=1) 
    predictions = [round(value) for value in yprob]
    print(f"predictions: {predictions}")  # model output is here !!!!!!!!!!!!!!!!!!!!!!!!!

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


if __name__ == "__main__":
    testing()
