import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchmetrics.classification import BinaryConfusionMatrix
from torchmetrics import Accuracy, AUROC, Recall, Specificity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def compute_cross_entropy(outputs, label):
    # Using F.cross_entropy directly, without one-hot encoding labels
    critiria = nn.BCELoss()
    cross_entropy = critiria(outputs.float(), F.one_hot(label, outputs.shape[-1]).float())

    return cross_entropy.item()

def compute_batch_metrics(outputs, label):
    if np.shape(outputs)[-1] == 2:
        task = 'binary'
    else:
        task = 'multiclass'

    accuracy_metric = Accuracy(task=task).to(device)
    recall_metric = Recall(task=task).to(device)
    specificity_metric = Specificity(task=task).to(device)
    confusion_matrix_metric = BinaryConfusionMatrix(task=task).to(device)
    outputs = torch.argmax(outputs, axis=1)
    metrics_dict = {
        'accuracy': accuracy_metric(outputs, label).item(),
        'recall': recall_metric(outputs, label).item(),
        'specificity': specificity_metric(outputs, label).item(),
        'confusion_matrix': confusion_matrix_metric(outputs, label).detach().cpu().numpy()
    }

    return metrics_dict

def calculate_classification_metrics(confusion_matrix):
    # Sum the confusion matrix across all steps in the epoch
    sum_confusion_matrix = torch.sum(torch.stack(confusion_matrix), dim=0)

    # Calculate metrics
    true_positive = sum_confusion_matrix[1, 1].item()
    false_positive = sum_confusion_matrix[0, 1].item()
    false_negative = sum_confusion_matrix[1, 0].item()
    true_negative = sum_confusion_matrix[0, 0].item()

    # Use torchmetrics to calculate metrics
    # auroc_metric = AUROC()

    # Calculate metrics
    recall = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    precision = true_positive / (true_positive + false_positive)
    f1_score = 2 * (precision * recall) / (precision + recall)
    # auc = auroc_metric(outputs[:, 1], label)  # Assuming outputs is a tensor with probability scores for the positive class

    metrics_dict = {
        'recall': recall.item(),
        'specificity': specificity.item(),
        'precision': precision,
        'f1_score': f1_score,
        # 'auc': auc.item(),
    }

    return metrics_dict

