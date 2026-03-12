from abc import abstractmethod
from typing import Callable, List, Optional

import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score


class Metric:
    NAME = 'metric'

    @abstractmethod
    def __call__(self, y_preds: np.ndarray, y_trues: np.ndarray) -> None:
        pass


class AccuracyMetric(Metric):
    NAME = 'accuracy'

    def __call__(self, y_preds: np.ndarray, y_trues: np.ndarray) -> float:
        """
        Args:
            y_preds (np.ndarray): The model outputs of the shape [N, C].
            y_trues (np.ndarray): The ground truth of the shape [N].
        """
        pred_classes = np.argmax(y_preds, axis=-1)
        accuracy = np.mean((y_trues == pred_classes).astype(np.float32))
        return accuracy


class BinaryAccuracyMetric(Metric):
    NAME = 'binary_accuracy'

    def __init__(
        self,
        threshold: float = 0.0,
    ):
        super().__init__()
        self.threshold = threshold

    def __call__(self, y_preds: np.ndarray, y_trues: np.ndarray) -> List[float]:
        """
        Args:
            y_preds (np.ndarray): The model outputs of the shape [N, C].
            y_trues (np.ndarray):
                The ground truth of the shape [N, C]. Set the value of DONTCARE as
                np.nan.
        """
        y_preds_bool = y_preds > self.threshold  # [N, C]
        y_trues_bools = y_trues.astype(bool)  # [N, C]
        correctness = y_preds_bool == y_trues_bools  # [N, C]

        dont_care_mask = np.isnan(y_trues)  # [N, C]
        masked_correctness = np.where(
            dont_care_mask,
            np.zeros_like(correctness),
            correctness,
        )
        binary_accuracies = (
            masked_correctness.astype(float).sum(axis=0) / np.logical_not(dont_care_mask).astype(float).sum(axis=0)
        ).tolist()  # [C]
        return binary_accuracies


class AUCMetric(Metric):
    NAME = 'auc'

    def __init__(
        self,
        activation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.activation_fn = activation_fn
        self.verbose = verbose

    def __call__(self, y_preds: np.ndarray, y_trues: np.ndarray) -> List[float]:
        """
        Args:
            y_preds (np.ndarray): The model outputs of the shape [N, C].
            y_trues (np.ndarray): The ground truth of the shape [N].
        """
        if self.activation_fn is not None:
            y_preds = self.activation_fn(y_preds)

        auc_list = []
        for channel_index in range(y_preds.shape[1]):
            y_true_binary = (y_trues == channel_index).astype(np.int32)
            y_pred_binary = y_preds[:, channel_index]

            if len(np.unique(y_true_binary)) == 2:
                auc = roc_auc_score(
                    y_true_binary,
                    y_pred_binary,
                )
            else:
                if self.verbose:
                    print(f'Unable to calculate the AUC for the {channel_index}-th class.')
                auc = np.nan
            auc_list.append(auc)

        return auc_list


class MultiLabelAUCMetric(Metric):
    NAME = 'multi_label_auc'

    def __init__(
        self,
        activation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.activation_fn = activation_fn
        self.verbose = verbose

    def __call__(self, y_preds: np.ndarray, y_trues: np.ndarray) -> List[float]:
        """
        Args:
            y_preds (np.ndarray): The model outputs of the shape [N, C].
            y_trues (np.ndarray):
                The ground truth of the shape [N, C]. Set the value of DONTCARE as
                np.nan.
        """
        if self.activation_fn is not None:
            y_preds = self.activation_fn(y_preds)

        auc_list = []
        for channel_index in range(y_preds.shape[1]):
            y_true_binary = y_trues[:, channel_index]
            y_pred_binary = y_preds[:, channel_index]

            dont_care_mask = np.isnan(y_true_binary)
            y_true_binary = y_true_binary[~dont_care_mask]
            y_pred_binary = y_pred_binary[~dont_care_mask]

            if len(np.unique(y_true_binary)) == 2:
                auc = roc_auc_score(
                    y_true_binary,
                    y_pred_binary,
                )
            else:
                if self.verbose:
                    print(f'Unable to calculate the AUC for the {channel_index}-th class.')
                auc = np.nan
            auc_list.append(auc)

        return auc_list


class ConcordanceIndexMetric(Metric):
    NAME = 'concordance_index'

    def __call__(self, y_preds: np.ndarray, y_trues: np.ndarray) -> float:
        """
        Args:
            y_preds (np.ndarray): The predicted risk scores of the shape [N, 1].
            y_trues (np.ndarray):
                A [N, 2] array with the survival statuses (0 for death, 1 for alive) and
                the last observed times.
        """
        # Shape checking
        if y_preds.shape[1] != 1:
            raise RuntimeError(f'The y_preds shape {y_preds} is invalid.')
        if y_trues.shape[1] != 2:
            raise RuntimeError(f'The y_trues shape {y_trues} is invalid.')

        # Aliasing
        predicted_risks = y_preds[:, 0]
        survival_statuses = y_trues[:, 0]
        last_observed_times = y_trues[:, 1]

        # Calculation
        cindex = concordance_index(
            event_times=last_observed_times,
            predicted_scores=(-predicted_risks),
            event_observed=(1 - survival_statuses),
        )
        return cindex
