from __future__ import annotations

import argparse
import typing as t
from collections import OrderedDict
from pathlib import Path

import devtools
import numpy as np
import pydantic
import tqdm
from sklearn.metrics import roc_auc_score

from ..components.config import TestConfig
from ..components.dataset import ConcatDataset, Dataset


class AreaStatistics(pydantic.BaseModel):
    mean: float
    min: float
    q1: float
    q2: float
    q3: float
    max: float

    @classmethod
    def from_list(cls: t.Type[AreaStatistics], areas: t.Sequence[int]) -> AreaStatistics:
        mean = float(np.mean(areas))
        min = float(np.min(areas).astype(float))
        q1 = float(np.quantile(areas, 0.25))
        q2 = float(np.quantile(areas, 0.5))
        q3 = float(np.quantile(areas, 0.75))
        max = float(np.max(areas))
        area_statistics = cls(
            mean=mean,
            min=min,
            q1=q1,
            q2=q2,
            q3=q3,
            max=max,
        )
        return area_statistics


class BinarizedHeatmapMetrics(pydantic.BaseModel):
    area_indicator_auc: float
    area_statistics_of_positive: AreaStatistics
    area_statistics_of_negative: AreaStatistics


class ClassHeatmapMetrics(pydantic.BaseModel):
    class_name: str
    by_threshold: t.OrderedDict[float, BinarizedHeatmapMetrics]


class HeatmapMetrics(pydantic.BaseModel):
    by_class: t.List[ClassHeatmapMetrics]


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
        help=('Path to store the metrics. Default: ' '${RESULT_DIR}/heatmap_metrics.json'),
    )
    parser.add_argument(
        '--thresholding-splits',
        type=int,
        default=100,
        help='The number of splits for thresholding.',
    )
    args = parser.parse_args()

    # Read the config
    config = TestConfig.from_yaml(args.config)

    # Set default values
    if args.metric_output is None:
        args.metric_output = Path(config.RESULT_DIR) / 'heatmap_metrics.json'
    if config.CLASS_NAMES is None:
        class_names = [f'class{idx}' for idx in range(config.NUM_CLASSES)]
    else:
        class_names = config.CLASS_NAMES

    # Read the test dataset
    test_subdatasets = [
        Dataset(
            csv_path=dataset_config.TEST_CSV_PATH,
            slide_dir=dataset_config.SLIDE_DIR,
            slide_file_extension=dataset_config.SLIDE_FILE_EXTENSION,
            contour_dir=dataset_config.CONTOUR_DIR,
            format='fit',
            resize_ratio=config.RESIZE_RATIO,
        )
        for dataset_config in config.TEST_DATASET_CONFIGS
    ]
    test_dataset = ConcatDataset(datasets=test_subdatasets)

    # Extract information
    slide_names = [test_dataset.get_slide_name(idx) for idx in range(len(test_dataset))]
    y_trues = []
    for idx in range(len(test_dataset)):
        y_true = test_dataset.get_y_true(idx)
        if isinstance(y_true, int):
            one_hot = np.zeros(shape=[config.NUM_CLASSES], dtype=int)
            one_hot[y_true] = 1
            y_true = one_hot
        y_trues.append(y_true)
    y_trues = np.array(y_trues)

    # Per-class evaluation
    class_heatmap_metrics_list: t.List[ClassHeatmapMetrics] = []
    for class_index, class_name in enumerate(class_names):
        print(f'Evaluating {class_name}...')

        # Ground-truths of this class
        y_trues_class = y_trues[:, class_index]

        # Read the heatmaps
        heatmaps_class = []
        for slide_name in tqdm.tqdm(slide_names, desc='Read heatmaps'):
            heatmap_path = Path(config.VIZ_RESULT_DIR) / f'{slide_name}.npy'
            heatmap = np.load(heatmap_path)[..., class_index]
            heatmap = np.nan_to_num(heatmap, nan=0.0)
            heatmaps_class.append(heatmap)

        # Iterate over thresholds
        binarized_heatmap_metrics_dict: t.OrderedDict[float, BinarizedHeatmapMetrics] = OrderedDict()
        for threshold in tqdm.tqdm(
            np.linspace(start=0.0, stop=1.0, num=args.thresholding_splits, endpoint=False)[1:],
            desc='Iterate thresholds',
        ):
            # Calculate the positive area
            areas_class_threshold = []
            for heatmap in heatmaps_class:
                binarized_heatmap = heatmap >= threshold
                area = binarized_heatmap.astype(int).sum()
                areas_class_threshold.append(area)
            areas_class_threshold = np.array(areas_class_threshold)

            # Calculate statistics
            is_nan = np.isnan(y_trues_class)
            area_indicator_auc = roc_auc_score(y_true=y_trues_class[~is_nan], y_score=areas_class_threshold[~is_nan])
            area_statistics_of_positive = AreaStatistics.from_list(areas_class_threshold[y_trues_class == 1])
            area_statistics_of_negative = AreaStatistics.from_list(areas_class_threshold[y_trues_class == 0])
            binarized_heatmap_metrics = BinarizedHeatmapMetrics(
                area_indicator_auc=area_indicator_auc,
                area_statistics_of_positive=area_statistics_of_positive,
                area_statistics_of_negative=area_statistics_of_negative,
            )
            binarized_heatmap_metrics_dict[threshold] = binarized_heatmap_metrics

        # Aggregate results
        class_heatmap_metrics = ClassHeatmapMetrics(class_name=class_name, by_threshold=binarized_heatmap_metrics_dict)
        class_heatmap_metrics_list.append(class_heatmap_metrics)

    heatmap_metrics = HeatmapMetrics(by_class=class_heatmap_metrics_list)

    # Output the metrics
    devtools.debug(heatmap_metrics)
    with open(args.metric_output, 'w') as file:
        file.write(heatmap_metrics.json(indent=4))


if __name__ == '__main__':
    main()
