#!/usr/bin/env python

import os
import csv
import random
import warnings

from sklearn import metrics
from itertools import product
from shutil import copyfile

import numpy as np

random.seed(0)
np.random.seed(0)

warnings.filterwarnings('ignore')

SUBMISSIONS_DIRECTORY_PATH = ''
GROUND_TRUTH_PATH = ''

RESULTS_DIRECTORY = ''

AVERAGING_TECHNIQUES = ['macro', 'micro']

def binary_confusion_matrix(y_true, y_pred, y_target):

    targ_tmp = np.where(y_true != y_target, 0, 1)
    pred_tmp = np.where(y_pred != y_target, 0, 1)

    class_labels = np.unique(np.concatenate((targ_tmp, pred_tmp)))

    z = list(zip(targ_tmp, pred_tmp))
    lst = [ z.count(combi) for combi in product(class_labels, repeat=2) ]

    mat = np.asarray(lst)[:, None].reshape(2, 2)

    return np.array(mat, dtype=np.float32)

def save_confusion_matrix(path, confusion_matrix, labels=None):
    
    confusion_list = confusion_matrix.astype(int).tolist()

    if labels is not None:
        for label_index, label in enumerate(labels):
            confusion_list[label_index].insert(0, label)
        confusion_list.insert(0, ['predicted/actual', *labels])

    with open(path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(confusion_list)

def read_submission(submission_path):
    results = {}
    duplicates = {}
    lines = []
    with open(submission_path) as f:
        for row in csv.reader(f):
            lines.append(row)
            image_id = os.path.splitext(row[0].strip())[0].strip()

            if image_id in results:
                # print(f'Duplicate in prediction: { image_id }')
                duplicates[image_id] = row[1].strip()
                continue
                
            results[image_id] = row[1].strip()
    return duplicates, np.array(lines), results

def read_gt(gt_path):
    gt_classes = {}
    gt_results = {}
    gt_classes_list = []
    with open(gt_path) as csv_data:
        class_id = 0
        for row in csv.reader(csv_data):
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

def evaluate_submission(submission_path):

    submission_attributes = os.path.basename(submission_filename).split('_')
    
    
    # Extract run attributes
    team_name = submission_attributes[1]
    task_name = submission_attributes[2]
    run_id    = os.path.splitext('_'.join(submission_attributes[3:]))[0]

    team_result_path = os.path.join(RESULTS_DIRECTORY, team_name, task_name, run_id)

    if not os.path.exists(team_result_path):
        os.makedirs(team_result_path)

    # Copy submission file into results file for easy access
    copyfile(submission_path, f'{ team_result_path }/{ os.path.basename(submission_path) }')
    
    # Read submission file
    duplicates, submission_lines, prediction_results = read_submission(submission_path)

    print(f'The number of predicted images is: { len(submission_lines) }')
    print(f'The number of unique predicted images is: { len(np.unique(submission_lines[:, 0])) }')
    print(f'The number of duplicates is : { len(duplicates) }')

    average_time = None
    minimum_time = None
    maximum_time = None

    # Check if prediction line contains time information
    if len(submission_lines[0]) > 3:
        submission_times = np.array(submission_lines[:, 3], dtype=np.float32)

        # Calculate time information
        average_time = np.mean(submission_times) * 1000
        minimum_time = np.min(submission_times)  * 1000
        maximum_time = np.max(submission_times)  * 1000

        average_FPS = 1.0 / (np.mean(submission_times))
        maximum_FPS = 1.0 / (np.min(submission_times))
        minimum_FPS = 1.0 / (np.max(submission_times))

    # Read the ground truth and extract classes
    gt_classes, gt_results = read_gt(GROUND_TRUTH_PATH)
    number_of_classes = len(gt_classes.items())

    # Check for prediction tuples. If present, select the correct prediction for evaluation
    if len(duplicates.items()) > 0:
        for image_id, prediction in duplicates.items():
            if not image_id in gt_results:
                continue
            if prediction == gt_results[image_id]:
                prediction_results[image_id] = prediction

    y_pred, y_truth = [], []
    missing = []

    # Initialize confusion matrix
    confusion_matrix = np.zeros((number_of_classes, number_of_classes), dtype=np.float32)

    # iterate over ground truth
    for image_id, actual_class in gt_results.items():
        
        # Check for missing predictions
        if image_id not in prediction_results:
            missing.append(image_id)
            continue

        # Get class id for both the actual and predicted class 
        actual_class_id = gt_classes[actual_class]
        predicted_class_id = gt_classes[prediction_results[image_id]]
        
        # Increment confusion matrix
        confusion_matrix[predicted_class_id, actual_class_id] += 1

        y_pred.append(predicted_class_id)
        y_truth.append(actual_class_id)

    y_pred = np.array(y_pred)
    y_truth = np.array(y_truth)

    if len(y_truth) != len(y_pred):
        raise Exception('The number of predicted values is NOT equal to the ground truth!')

    # Write missing predictions to file in team results
    if len(missing) != 0:
        print(f'Submission missing the prediction of { len(missing) } images!')
        with open(os.path.join(team_result_path, 'missing_predictions.txt'), 'w') as f:
            for image in missing:
                f.write(f'{ image }\n')

    # Save confusion matrix to team results
    save_confusion_matrix(os.path.join(team_result_path, f'confusion_matrix.csv'),
        confusion_matrix, labels=[ el[0] for el in sorted(gt_classes.items(), key=lambda x: x[1]) ])

    # Calculate metrics for the individual classes
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    PREC = metrics.precision_score(y_truth, y_pred, average=None)
    REC  = metrics.recall_score(y_truth, y_pred, average=None)
    F1   = metrics.f1_score(y_truth, y_pred, average=None)
    SPEC = TN / (TN + FP)

    # Create directories for class specific metrics
    class_specific_metrics_path   = os.path.join(team_result_path, 'class_specific_metrics')
    class_specific_confusion_path = os.path.join(team_result_path, 'class_specific_confusion_matricies')

    if not os.path.exists(class_specific_metrics_path):
        os.makedirs(class_specific_metrics_path)

    if not os.path.exists(class_specific_confusion_path):
        os.makedirs(class_specific_confusion_path)

    # Save metrics for individual classes
    for class_name, class_id in gt_classes.items():
        
        save_confusion_matrix(os.path.join(class_specific_confusion_path, f'{ class_name }_confusion_matrix.csv'),
            binary_confusion_matrix(y_truth, y_pred, class_id), labels=[f'non-{ class_name }', class_name])

        with open(os.path.join(class_specific_metrics_path, f'{ class_name }_metrics.csv'), 'w') as f:
            f.write(f'metric,value\n')
            f.write(f'true-positives,{ int(TP[class_id]) }\n')
            f.write(f'true-negatives,{ int(TN[class_id]) }\n')
            f.write(f'false-positives,{ int(FP[class_id]) }\n')
            f.write(f'false-negatives,{ int(FN[class_id]) }\n')
            f.write(f'precision,{ PREC[class_id] }\n')
            f.write(f'recall/sensitivity,{ REC[class_id] }\n')
            f.write(f'specificity,{ SPEC[class_id] }\n')
            f.write(f'f1,{ F1[class_id] }\n')

    # Calculate specificity seperateley as it is not avaiable in Scikit-Learn
    SPEC = { 'macro': np.mean(TN / (TN + FP)), 'micro': np.mean(sum(TN) / (sum(TN) + sum(FP))) }

    # Calculate multi-class results using micro and macro averaging
    for average_technique in AVERAGING_TECHNIQUES:

        F1   = metrics.f1_score(y_truth, y_pred, average=average_technique)
        PREC = metrics.precision_score(y_truth, y_pred, average=average_technique)
        REC  = metrics.recall_score(y_truth, y_pred, average=average_technique)
        MCC  = metrics.matthews_corrcoef(y_truth, y_pred)
        
        # Save results to team results
        with open(os.path.join(team_result_path, f'{ average_technique }_average_metrics.csv'), 'w') as f:
            f.write(f'metric,value\n')
            f.write(f'true-positives,{ int(sum(TP)) }\n')
            f.write(f'true-negatives,{ int(sum(TN)) }\n')
            f.write(f'false-positives,{ int(sum(FP)) }\n')
            f.write(f'false-negatives,{ int(sum(FN)) }\n')
            f.write(f'precision,{ PREC }\n')
            f.write(f'recall/sensitivity,{ REC }\n')
            f.write(f'specificity,{ SPEC[average_technique] }\n')
            f.write(f'f1,{ F1 }\n')
            f.write(f'mcc,{ MCC }\n')

            if average_time is not None:
                f.write(f'average-time,{ average_time }\n')
                f.write(f'minimum-time,{ minimum_time }\n')
                f.write(f'maximum-time,{ maximum_time }\n')
                f.write(f'average-FPS,{ average_FPS }\n')
                f.write(f'minimum-FPS,{ minimum_FPS }\n')
                f.write(f'maximum-FPS,{ maximum_FPS }\n')
        
        # Save results to common results
        with open(os.path.join(RESULTS_DIRECTORY, f'all_{ average_technique }_average_metrics.csv'), 'a') as f:
            f.write(f'{ team_name },{ task_name },{ run_id }')
            f.write(f',{ int(sum(TP)) },{ int(sum(TN)) },{ int(sum(FP)) },{ int(sum(FN)) }')
            f.write(f',{ PREC },{ REC },{ SPEC[average_technique] },{ F1 },{ MCC }')
            if average_time is not None:
                f.write(f',{ average_time },{ minimum_time },{ maximum_time },{ average_FPS },{ minimum_FPS },{ maximum_FPS }')
            f.write(f'\n')
        
if __name__ == '__main__':

    for average_technique in AVERAGING_TECHNIQUES:
        with open(os.path.join(RESULTS_DIRECTORY, f'all_{ average_technique }_average_metrics.csv'), 'w') as f:
            f.write('team-name,task-name,run-id,true-positives,true-negatives,false-positives,false-negatives,precision,recall/sensitivity,specificity,f1,mcc,average-time,minimum-time,maximum-time,average-FPS,minimum-FPS,maximum-FPS\n')

    for submission_filename in os.listdir(SUBMISSIONS_DIRECTORY_PATH):

        if not os.path.splitext(submission_filename)[1] == '.csv':
            continue

        print(f'Evaluating { submission_filename }...')

        evaluate_submission(os.path.join(SUBMISSIONS_DIRECTORY_PATH, submission_filename))