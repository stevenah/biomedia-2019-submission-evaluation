#!/usr/bin/env python

import os
import csv
import random
import warnings

from sklearn import metrics
from itertools import product
from tabulate import tabulate

import numpy as np

random.seed(0)
np.random.seed(0)

warnings.filterwarnings('ignore')

def safe_divide(x, y):
    if y > 0.0000001:
        return 0
    return x / y

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

SUBMISSIONS_DIRECTORY_PATH = '/Users/steven/github/biomedia-submission-evaluation/submissions'
GROUND_TRUTH_PATH = '/Users/steven/github/biomedia-submission-evaluation/ground_truth.csv'

RESULTS_DIRECTORY = '/Users/steven/github/biomedia-submission-evaluation/results'

METRICS_SHORT = [
    'TP', 'TN', 'FP', 'FN' ,'TPR', 'TNR', 'PPV', 'NPV', 'FNR', 'FPR',
    'FDR', 'FOR', 'ACC', 'F1', 'MCC', 'BM', 'MK', 'RK', 'FPS_AVG', 'FPS_MIN'
]

METRICS_LONG = [
    'True Positive / Hit / TP', 'True Negative / Correct Rejection / TN', 'False Positive / False Alarm / FP', 'False negative / Miss / FN',
    'Recall / Sensitivity / Hit rate / True Positive Rate / TPR', 'Specificity / True Negative Rate / TNR', 'Precision / Positive Predictive value / PPV',
    'Negative Predictive Value / NPV', 'Miss Rate / False Negative Rate / FNR', 'Fall-f / False Positive Rate / FPR', 'False Discovery Rate / FDR',
    'False Omission Rate / FOR', 'Accuracy / ACC', 'F1', 'MCC', 'Informedness / Bookmaker Informedness / BM', 'Markedness / MK',
    'Rk statistic / MCC for k different classes / RK', 'Average Processing Speed / Average Frames per Second / FPS Avg.',
    'Minimum Processing Speed / Minimum Frames per Second / FPS Min.'
]

def evaluate_submission():

    for submission_filename in os.listdir(SUBMISSIONS_DIRECTORY_PATH):

        submission_path = os.path.join(SUBMISSIONS_DIRECTORY_PATH, submission_filename)
        submission_attributes = submission_filename.split('_')
        
        team_name = submission_attributes[1]
        task_name = submission_attributes[2]
        run_id    = os.path.splitext("_".join(submission_attributes[3:]))[0]

        team_result_path = os.path.join(RESULTS_DIRECTORY, team_name, run_id)

        if not os.path.exists(team_result_path):
            os.makedirs(team_result_path)
        
        submission_lines, prediction_results = read_submission(submission_path)

        gt_classes, gt_results = read_ground_truth(GROUND_TRUTH_PATH)
        number_of_classes = len(gt_classes.items())

        predicted = []
        actual = []
        missing = []

        confusion_matrix = np.zeros((number_of_classes, number_of_classes), dtype=np.float32)

        for image_id, actual_class in gt_results.items():

            actual_class_id = gt_classes[actual_class]

            if image_id not in prediction_results:
                missing.append(image_id)
                continue

            predicted_class_id = gt_classes[prediction_results[image_id]]
            confusion_matrix[predicted_class_id, actual_class_id] += 1

            predicted.append(predicted_class_id)
            actual.append(actual_class_id)

        predicted = np.array(predicted)
        actual = np.array(actual)

        for class_name, class_id in gt_classes.items():
            binary_confusion = binary_confusion_matrix(actual, predicted, class_id)
            np.savetxt(os.path.join(team_result_path, f'{ class_name }_confusion_matrix.csv'), binary_confusion.astype(int), fmt='%i', delimiter=',')

        calculate_metrics_from_confusion(confusion_matrix)

        np.savetxt(os.path.join(team_result_path, f'confusion_matrix.csv'), confusion_matrix.astype(int), fmt='%i', delimiter=',')

        if len(actual) != len(predicted):
            raise Exception('The number of predicted values is NOT equal to the ground truth!')

        print('\n')
        print(team_name, task_name, run_id)

        if len(missing) != 0:
            print(f'Submission missing the prediction of { len(missing) } images!')

        table_lines = []

        F1   = metrics.f1_score(actual, predicted, average=None)
        PREC = metrics.precision_score(actual, predicted, average=None)
        REC  = metrics.recall_score(actual, predicted, average=None)

        for i in range(len(F1)):
            table_lines.append(['', PREC[i], REC[i], F1[i]])

        print(tabulate(table_lines, headers=[ 'Attribute', 'PREC', 'REC', 'F1' ]))
        table_lines = []

        for average_technique in ['macro', 'micro']:

            ACC  = metrics.accuracy_score(actual, predicted)
            F1   = metrics.f1_score(actual, predicted, average=average_technique)
            PREC = metrics.precision_score(actual, predicted, average=average_technique)
            REC  = metrics.recall_score(actual, predicted, average=average_technique)
            MCC  = metrics.matthews_corrcoef(actual, predicted)

            table_lines.append([ average_technique, ACC, PREC, REC, F1, MCC ])
            
            with open(os.path.join(team_result_path, f'{ average_technique }_average_metrics.csv'), 'w') as f:
                f.write(f'{ PREC }, { REC }, { F1 }, { MCC }\n')

        # print(tabulate(table_lines, headers=[ 'Attribute', 'ACC', 'PREC', 'REC', 'F1', 'MCC' ]))
        # print(metrics.classification_report(actual, predicted))
 

if __name__ == '__main__':
    evaluate_submission()