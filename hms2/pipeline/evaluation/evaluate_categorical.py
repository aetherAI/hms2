from __future__ import annotations

import argparse
import typing as t
from collections import OrderedDict
from pathlib import Path

import devtools
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pydantic
import sklearn.metrics

from ..components.config import TestConfig
from ..components.losses import get_label_type
from ..components.utils import TestResults
from .utils import BinaryMetrics


class ConfusionMatrix(pydantic.BaseModel):
    titles: t.List[str]
    confusion_matrix: t.List[t.List[float]]


class OverallCategoricalMetrics(pydantic.BaseModel):
    accuracy: float = np.nan
    confusion_matrix: ConfusionMatrix


class ClassCategoricalMetrics(pydantic.BaseModel):
    title: str
    by_max: BinaryMetrics
    by_threshold: t.OrderedDict[float, BinaryMetrics]


class CategoricalMetrics(pydantic.BaseModel):
    overall: OverallCategoricalMetrics
    by_class: t.List[ClassCategoricalMetrics]


def main() -> None:
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Config file',
    )
    parser.add_argument(
        '--metric-output',
        type=Path,
        default=None,
        help=('Path to store the metrics. Default: ' '${RESULT_DIR}/categorical_metrics.json'),
    )
    parser.add_argument(
        '--confusion-matrix-output',
        type=Path,
        default=None,
        help=('Folder to store the ROC curves. Default: ' '${RESULT_DIR}/confusion_matrix.pdf'),
    )
    parser.add_argument(
        '--thresholding-splits',
        type=int,
        default=20,
        help='The number of splits for thresholding.',
    )
    args = parser.parse_args()

    # Read the config
    config = TestConfig.from_yaml(args.config)
    label_type = get_label_type(config.LOSS)
    if label_type != 'multi_class':
        raise ValueError(
            'Only a categorical classification model is applicable to this script.',
        )

    # Set default values
    if args.metric_output is None:
        args.metric_output = Path(config.RESULT_DIR) / 'categorical_metrics.json'
    if args.confusion_matrix_output is None:
        args.confusion_matrix_output = Path(config.RESULT_DIR) / 'confusion_matrix.pdf'
    if config.CLASS_NAMES is None:
        titles = [f'class{idx}' for idx in range(config.NUM_CLASSES)]
    else:
        titles = config.CLASS_NAMES

    # Read the test results
    test_results = TestResults.open(config.TEST_RESULT_PATH)
    _, y_trues, y_preds = zip(*test_results)
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    y_pred_indices = y_preds.argmax(axis=-1)

    # Derive a confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_true=y_trues,
        y_pred=y_pred_indices,
        labels=range(config.NUM_CLASSES),  # In case some classes do not present.
    )

    # Analyze the confusion matrix
    class_categorical_metrics: t.List[ClassCategoricalMetrics] = []
    for class_index in range(config.NUM_CLASSES):
        title = titles[class_index]
        by_max_metrics = BinaryMetrics.from_predictions(
            y_true=(y_trues == class_index),
            y_pred=(y_pred_indices == class_index),
        )

        by_threshold_metrics: t.OrderedDict[float, BinaryMetrics] = OrderedDict()
        for threshold in np.linspace(
            start=0.0,
            stop=1.0,
            num=args.thresholding_splits,
            endpoint=False,
        )[1:]:
            metrics = BinaryMetrics.from_predictions(
                y_true=(y_trues == class_index),
                y_pred=(y_preds[:, class_index] > threshold),
            )
            by_threshold_metrics[threshold] = metrics

        metrics = ClassCategoricalMetrics(
            title=title,
            by_max=by_max_metrics,
            by_threshold=by_threshold_metrics,
        )
        class_categorical_metrics.append(metrics)

    # Output the metrics
    accuracy = sklearn.metrics.accuracy_score(y_true=y_trues, y_pred=y_pred_indices)
    metrics = CategoricalMetrics(
        overall=OverallCategoricalMetrics(
            accuracy=accuracy,
            confusion_matrix=ConfusionMatrix(
                titles=titles,
                confusion_matrix=confusion_matrix.tolist(),
            ),
        ),
        by_class=class_categorical_metrics,
    )
    devtools.debug(metrics)
    with open(args.metric_output, 'w') as file:
        file.write(metrics.json(indent=4))

    # Plot the confusion matrix
    plt.figure()
    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=y_trues,
        y_pred=y_pred_indices,
        labels=range(config.NUM_CLASSES),
        display_labels=titles,
        cmap=_get_color_map(),
        colorbar=False,
    )
    plt.savefig(args.confusion_matrix_output)


def _get_color_map() -> matplotlib.colors.LinearSegmentedColormap:
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        'gigapixel',
        ['#F5FAFF', '#0074FF'],
    )
    return color_map


if __name__ == '__main__':
    main()
