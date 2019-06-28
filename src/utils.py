import os
import csv
import random
import warnings

from sklearn import metrics
from itertools import product
from tabulate import tabulate

import numpy as np

def safe_divide_np(x, y):
    return np.divide(x, y, out=np.zeros_like(x), where=y!=0)

def binary_confusion_matrix(y_true, y_pred, y_target):

    targ_tmp = np.where(y_true != y_target, 0, 1)
    pred_tmp = np.where(y_pred != y_target, 0, 1)

    class_labels = np.unique(np.concatenate((targ_tmp, pred_tmp)))

    z = list(zip(targ_tmp, pred_tmp))
    lst = [ z.count(combi) for combi in product(class_labels, repeat=2) ]

    mat = np.asarray(lst)[:, None].reshape(2, 2)

    return np.array(mat, dtype=np.float32)

def read_submission(submission_path):
    results = {}
    lines = []
    with open(submission_path) as f:
        data = csv.reader(f)
        for row in data:
            lines.append(row)
            image_id = os.path.splitext(row[0].strip())[0].strip()
            class_name = row[1].strip()
            results[image_id] = class_name
            if len(row) > 3:
                pass # Calcualte time
    return np.array(lines), results

def calculate_metrics_from_confusion(confusion_matrix):

        FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        TP = np.diag(confusion_matrix)
        TN = confusion_matrix.sum() - (FP + FN + TP)

        TPR = safe_divide_np(TP, (TP + FN))
        TNR = safe_divide_np(TN, (TN + FP))
        PPV = safe_divide_np(TP, (TP + FP))
        NPV = safe_divide_np(TN, (TN + FN))
        FNR = safe_divide_np(FN, (FN + TP))
        FPR = safe_divide_np(FP, (FP + TN))
        FDR = safe_divide_np(FP, (FP + TP))
        FOR = safe_divide_np(FN, (FN + TN))
        ACC = safe_divide_np((TP + TN), (TP + TN + FP + FN))
        F1  = safe_divide_np(2 * TP, (2 * TP + FP + FN))
        MCC = safe_divide_np((TP * TN - FP * FN), np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))

        BM  = TPR + TNR - 1
        MK  = PPV + NPV - 1

def read_ground_truth(gt_path):

    gt_classes = {}
    gt_results = {}
    gt_classes_list = []

    with open(gt_path) as csv_data:
        data = csv.reader(csv_data)
        class_id = 0
        
        for row in data:
            image_id = os.path.splitext(row[0].strip())[0].strip()
            if image_id in gt_results:
                print(f'Duplicate in GT: { image_id }')
                
            class_name = row[1].strip()
            gt_results[image_id] = class_name
            
            if not class_name in gt_classes_list:
                gt_classes_list.append(class_name)
                gt_classes[class_name] = class_id
                class_id += 1
                
    return gt_classes, gt_results