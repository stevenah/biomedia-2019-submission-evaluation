#!/usr/bin/env python

import os
import csv
import random
import warnings

from sklearn import metrics
from itertools import product
from tabulate import tabulate

import numpy as np

from utils import read_ground_truth, read_submission, binary_confusion_matrix

random.seed(0)
np.random.seed(0)

warnings.filterwarnings('ignore')

SUBMISSIONS_DIRECTORY_PATH = '/Users/steven/github/biomedia-2019-submission-evaluation/submissions'
GROUND_TRUTH_PATH = '/Users/steven/github/biomedia-2019-submission-evaluation/ground_truth.csv'

RESULTS_DIRECTORY = '/Users/steven/github/biomedia-2019-submission-evaluation/results'

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

def save_confusion_matrix(path, confusion_matrix):
    np.savetxt(path, confusion_matrix.astype(int), fmt='%i', delimiter=',')

def evaluate_submission(submission_path):

    submission_attributes = os.path.basename(submission_filename).split('_')
    
    team_name = submission_attributes[1]
    task_name = submission_attributes[2]
    run_id    = os.path.splitext('_'.join(submission_attributes[3:]))[0]

    team_result_path = os.path.join(RESULTS_DIRECTORY, team_name, task_name, run_id)

    average_time = None
    minimum_time = None
    maximum_time = None

    if not os.path.exists(team_result_path):
        os.makedirs(team_result_path)
    
    submission_lines, prediction_results = read_submission(submission_path)

    if len(submission_lines[0]) > 3:
        submission_times = np.array(submission_lines[:, 3], dtype=np.float32)
        average_time = np.mean(submission_times)
        minimum_time = np.min(submission_times)
        maximum_time = np.max(submission_times)

    gt_classes, gt_results = read_ground_truth(GROUND_TRUTH_PATH)
    number_of_classes = len(gt_classes.items())

    y_pred, y_truth = [], []
    missing = []

    confusion_matrix = np.zeros((number_of_classes, number_of_classes), dtype=np.float32)

    for image_id, actual_class in gt_results.items():

        actual_class_id = gt_classes[actual_class]

        if image_id not in prediction_results:
            missing.append(image_id)
            continue

        predicted_class_id = gt_classes[prediction_results[image_id]]
        confusion_matrix[predicted_class_id, actual_class_id] += 1

        y_pred.append(predicted_class_id)
        y_truth.append(actual_class_id)

    y_pred = np.array(y_pred)
    y_truth = np.array(y_truth)

    if len(y_truth) != len(y_pred):
        raise Exception('The number of predicted values is NOT equal to the ground truth!')

    if len(missing) != 0:
        print(f'Submission missing the prediction of { len(missing) } images!')

    save_confusion_matrix(os.path.join(team_result_path, f'confusion_matrix.csv'), confusion_matrix)

    PREC = metrics.precision_score(y_truth, y_pred, average=None)
    REC  = metrics.recall_score(y_truth, y_pred, average=None)
    F1   = metrics.f1_score(y_truth, y_pred, average=None)

    class_specific_metrics_path   = os.path.join(team_result_path, 'class_specific_metrics')
    class_specific_confusion_path = os.path.join(team_result_path, 'class_specific_confusion_matricies')

    if not os.path.exists(class_specific_metrics_path):
        os.makedirs(class_specific_metrics_path)

    if not os.path.exists(class_specific_confusion_path):
        os.makedirs(class_specific_confusion_path)

    for class_name, class_id in gt_classes.items():
        
        save_confusion_matrix(os.path.join(class_specific_confusion_path, f'{ class_name }_confusion_matrix.csv'),
            binary_confusion_matrix(y_truth, y_pred, class_id))

        with open(os.path.join(class_specific_metrics_path, f'{ class_name }_metrics.csv'), 'a') as f:
            f.write(f'{ PREC[class_id] },{ REC[class_id] },{ F1[class_id] }\n')

    for average_technique in ['macro', 'micro']:

        ACC  = metrics.accuracy_score(y_truth, y_pred)
        F1   = metrics.f1_score(y_truth, y_pred, average=average_technique)
        PREC = metrics.precision_score(y_truth, y_pred, average=average_technique)
        REC  = metrics.recall_score(y_truth, y_pred, average=average_technique)
        MCC  = metrics.matthews_corrcoef(y_truth, y_pred)
        
        with open(os.path.join(team_result_path, f'{ average_technique }_average_metrics.csv'), 'w') as f:
            f.write(f'{ PREC },{ REC },{ F1 },{ MCC }\n')

if __name__ == '__main__':

    for submission_filename in os.listdir(SUBMISSIONS_DIRECTORY_PATH):

        if not os.path.splitext(submission_filename)[1] == '.csv':
            continue

        evaluate_submission(os.path.join(SUBMISSIONS_DIRECTORY_PATH, submission_filename))