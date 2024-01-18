from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics.metric import Metric
from torchmetrics import Accuracy, Precision, Recall, F1Score, Specificity, AUROC
import sys
sys.path.append('../')


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("all_met", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, metric: torch.Tensor, quantity: torch.Tensor):  # type: ignore

        self.all_met += metric
        self.total += quantity

    def compute(self):
        return self.all_met.float() / self.total


class MriModule(LightningModule):
    
    def __init__(
        self,
        num_log_images: int = 16,
        task: str = 'binary'
    ):
        """
        Number of images to log in tensorboard. Defaults to 16.

        Args:
            num_log_images (int, optional): Defaults to 16.
        """
        super().__init__()
        
        self.num_log_images = num_log_images
        self.task = task
        self.val_log_indices = None
        self.validation_step_outputs = []
        self.test_log_indices = None
        self.test_step_outputs = []
        # Define classification metrics from torchmetrics
        self.accuracy = Accuracy(task=self.task)
        self.precision = Precision(task=self.task)  # Assuming binary classification
        self.recall = Recall(task=self.task)
        self.f1 = F1Score(task=self.task)
        self.specificity = Specificity(task=self.task)
        self.AUC = AUROC(task=self.task)


        # Distributed Metric Sums
        self.val_accuracy_sum = DistributedMetricSum()
        self.val_precision_sum = DistributedMetricSum()
        self.val_recall_sum = DistributedMetricSum()
        self.val_f1_sum = DistributedMetricSum()
        self.val_specificity_sum = DistributedMetricSum()
        self.val_AUC_sum = DistributedMetricSum()
    
    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, self.global_step)

    def on_validation_epoch_end(self):
        all_predictions = []
        all_labels = []
        all_losses = []
        for val_log in self.validation_step_outputs:
            predictions = val_log["predictions"].cpu().numpy()
            labels = val_log["labels"].cpu().numpy()
            loss = val_log["loss"].cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_losses.append(loss)
        pred = np.argmax(all_predictions, axis=1)
        binary_predictions = np.array(pred)
        all_labels = np.array(all_labels)
        quantity = 1
        # print(np.size(all_labels),binary_predictions)
        # Compute metrics
        accuracy = self.accuracy(torch.tensor(binary_predictions), torch.tensor(all_labels))
        precision = self.precision(torch.tensor(binary_predictions), torch.tensor(all_labels))
        recall = self.recall(torch.tensor(binary_predictions), torch.tensor(all_labels))
        f1 = self.f1(torch.tensor(binary_predictions), torch.tensor(all_labels))
        specificity = self.specificity(torch.tensor(binary_predictions), torch.tensor(all_labels))
        AUC = self.AUC(torch.tensor(binary_predictions), torch.tensor(all_labels))
        # print(accuracy)

        # Update distributed metric sums
        self.val_accuracy_sum.update(accuracy,quantity)
        self.val_precision_sum.update(precision,quantity)
        self.val_recall_sum.update(recall,quantity)
        self.val_f1_sum.update(f1,quantity)
        self.val_specificity_sum.update(specificity,quantity)
        self.val_AUC_sum.update(AUC, quantity)


        self.log("val_accuracy", self.val_accuracy_sum.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision_sum.compute(), prog_bar=True)
        self.log("val_recall", self.val_recall_sum.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1_sum.compute(), prog_bar=True)
        self.log("val_specificity", self.val_specificity_sum.compute(), prog_bar=True)
        self.log("val_AUC", self.val_AUC_sum.compute(), prog_bar=True)
        # Log average loss
        avg_loss = np.mean(all_losses)
        self.log("validation_loss", avg_loss, prog_bar=True)

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        all_predictions = []
        all_labels = []
        all_losses = []

        for result in self.test_step_outputs:
            predictions = result["predictions"].cpu().numpy()
            labels = result["labels"].cpu().numpy()
            loss = result["loss"].cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_losses.append(loss)


        pred = np.argmax(all_predictions, axis=1)
        binary_predictions = np.array(pred)
        all_labels = np.array(all_labels)



        # Compute metrics
        accuracy = self.accuracy(torch.tensor(binary_predictions), torch.tensor(all_labels))
        precision = self.precision(torch.tensor(binary_predictions), torch.tensor(all_labels))
        recall = self.recall(torch.tensor(binary_predictions), torch.tensor(all_labels))
        f1 = self.f1(torch.tensor(binary_predictions), torch.tensor(all_labels))
        specificity = self.specificity(torch.tensor(binary_predictions), torch.tensor(all_labels))
        AUC = self.AUC(torch.tensor(binary_predictions), torch.tensor(all_labels))
        # Log metrics
        self.log("test_accuracy", accuracy, prog_bar=True)
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_recall", recall, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_specificity", specificity, prog_bar=True)
        self.log("test_AUC", AUC, prog_bar=True)

        # Log average loss
        avg_loss = np.mean(all_losses)
        self.log("test_loss", avg_loss, prog_bar=True)


        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_log_images", type=int, default=16, help="Number of images to log")
        parser.add_argument("--task", type=str, default='binary', help="Classification task")
        return parser