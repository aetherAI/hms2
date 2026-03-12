from __future__ import annotations

import argparse
import typing as t
from collections import OrderedDict
from pathlib import Path

import devtools
import numpy as np
import pydantic

from ..components.config import TestConfig
from ..components.losses import get_label_type
from ..components.utils import TestResults
from .utils import BinaryMetrics


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
        help=('Path to store the metrics. Default: ' '${RESULT_DIR}/multilabel_metrics.json'),
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
    if label_type != 'multi_label':
        raise ValueError(
            'Only a multi-label classification model is applicable to this script.',
        )

    # Set default values
    if args.metric_output is None:
        args.metric_output = Path(config.RESULT_DIR) / 'multilabel_metrics.json'
    if config.CLASS_NAMES is None:
        titles = [f'class{idx}' for idx in range(config.NUM_CLASSES)]
    else:
        titles = config.CLASS_NAMES

    # Read the test results
    test_results = TestResults.open(config.TEST_RESULT_PATH)
    _, y_trues, y_preds = zip(*test_results)
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)

    # Analyze the confusion matrix
    by_class_metrics: t.List[ByClassMetrics] = []
    for class_index in range(config.NUM_CLASSES):
        title = titles[class_index]

        # Remove don't care
        is_dont_care = np.isnan(y_trues[:, class_index])
        y_trues_of_class = y_trues[:, class_index][~is_dont_care]
        y_preds_of_class = y_preds[:, class_index][~is_dont_care]

        by_threshold_metrics: t.OrderedDict[float, BinaryMetrics] = OrderedDict()
        for threshold in np.linspace(
            start=0.0,
            stop=1.0,
            num=args.thresholding_splits,
            endpoint=False,
        )[1:]:
            metrics = BinaryMetrics.from_predictions(
                y_true=y_trues_of_class.astype(bool),
                y_pred=(y_preds_of_class > threshold),
            )
            by_threshold_metrics[threshold] = metrics

        metrics = ByClassMetrics(
            title=title,
            by_threshold=by_threshold_metrics,
        )
        by_class_metrics.append(metrics)

    # Output the metrics
    metrics = MultilabelMetrics(
        by_class=by_class_metrics,
    )
    devtools.debug(metrics)
    with open(args.metric_output, 'w') as file:
        file.write(metrics.json(indent=4))


class MultilabelMetrics(pydantic.BaseModel):
    by_class: t.List[ByClassMetrics]


class ByClassMetrics(pydantic.BaseModel):
    title: str
    by_threshold: t.OrderedDict[float, BinaryMetrics]


MultilabelMetrics.update_forward_refs()


if __name__ == '__main__':
    main()
