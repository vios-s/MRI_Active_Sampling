import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics import Accuracy, Precision, Recall, F1, Specificity, AUROC

def compute_cross_entropy(outputs, labels):
    # Using F.cross_entropy directly, without one-hot encoding labels
    cross_entropy = F.cross_entropy(outputs, labels)

    return cross_entropy



def compute_metrics(outputs, label):
    accuracy_metric = Accuracy()
    precision_metric = Precision()
    recall_metric = Recall()
    f1_metric = F1()
    specificity_metric = Specificity()
    auroc_metric = AUROC(pos_label=1)  # Assuming positive label is 1

    metrics_dict = {
        'accuracy': accuracy_metric(outputs, label).item(),
        'auc': auroc_metric(outputs, label),
        'precision': precision_metric(outputs, label).item(),
        'recall': recall_metric(outputs, label),
        'f1': f1_metric(outputs, label).item(),
        'specificity': specificity_metric(outputs, label),
    }

    return metrics_dict

