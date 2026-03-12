import argparse
import traceback
import typing as t
import warnings
from pathlib import Path

import matplotlib.cm
import numpy as np
import pydantic

from ..components.config import TestConfig
from ..components.losses import get_label_type
from ..components.utils import TestResults


try:
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    from rpy2.rinterface_lib.embedded import RRuntimeError
except ImportError:
    raise ImportError('You should install `hms2` with an extra `rpy2`.')


class ClassROCMetrics(pydantic.BaseModel):
    title: str
    auc: float = np.nan
    ci_low: float = np.nan
    ci_high: float = np.nan
    p_value: float = np.nan
    roc_curve: t.List[t.Tuple[float, float]] = []  # A list of (FPR, TPR).

    @property
    def legend(self) -> str:
        return f'{self.title} (AUC={self.auc:.4f} [{self.ci_low:.4f}-{self.ci_high:.4f}] ' f'P={self.p_value:.3f})'


class OverallROCMetrics(pydantic.BaseModel):
    macro_average_auc: float = np.nan


class ROCMetrics(pydantic.BaseModel):
    overall: OverallROCMetrics
    by_class: t.List[ClassROCMetrics]


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
        help='Path to store the metrics. Default: ${RESULT_DIR}/roc_metrics.json',
    )
    parser.add_argument(
        '--roc-output',
        type=Path,
        default=None,
        help='Folder to store the ROC curves. Default: ${RESULT_DIR}/rocs',
    )
    args = parser.parse_args()

    # Read the config
    config = TestConfig.from_yaml(args.config)
    label_type = get_label_type(config.LOSS)

    # Set default values
    if args.metric_output is None:
        args.metric_output = Path(config.RESULT_DIR) / 'roc_metrics.json'
    if args.roc_output is None:
        args.roc_output = Path(config.RESULT_DIR) / 'rocs'

    args.roc_output.mkdir(parents=True, exist_ok=True)

    # Read the test results
    test_results = TestResults.open(config.TEST_RESULT_PATH)
    _, y_trues, y_preds = zip(*test_results)
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)

    # Calculate the metrics per class
    class_roc_metrics_list: t.List[ClassROCMetrics] = []
    for class_index in range(config.NUM_CLASSES):
        title = _get_legend_title_from_list(config.CLASS_NAMES, class_index)
        try:
            roc = _get_roc(y_trues, y_preds, class_index, label_type)
            auc = roc[roc.names.index('auc')][0]
            ci_low = roc[roc.names.index('ci')][0]
            ci_high = roc[roc.names.index('ci')][2]
            test_result = _test_roc_with_null_hypothesis(roc)
            p_value = test_result[test_result.names.index('p.value')][0]
            roc_curve = _get_roc_curve(roc)
            class_roc_metrics = ClassROCMetrics(
                title=title,
                auc=auc,
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=p_value,
                roc_curve=roc_curve,
            )
            print(class_roc_metrics.legend)
        except RRuntimeError:
            traceback.print_exc()
            class_roc_metrics = ClassROCMetrics(title=title)
            print(f'Calculating the AUC of {title} failed.')

        class_roc_metrics_list.append(class_roc_metrics)

    # Calculate the overall metrics
    macro_average_auc = np.array([metric.auc for metric in class_roc_metrics_list if not np.isnan(metric.auc)]).mean()
    print(f'Macro-average AUC: {macro_average_auc:.4f}')

    # Save the metrics
    roc_metrics = ROCMetrics(
        overall=OverallROCMetrics(
            macro_average_auc=macro_average_auc,
        ),
        by_class=class_roc_metrics_list,
    )
    with open(args.metric_output, 'w') as file:
        file.write(roc_metrics.json(indent=4))
    print(f'The metrics have been saved as {args.metric_output} .')

    # Plot all-in-one ROC if required
    r_package_grdevices = _install_and_load_r_package('grDevices')
    roc_path = str(args.roc_output / 'roc.pdf')
    r_package_grdevices.pdf(file=roc_path)

    legends = []
    colors = []
    for class_index in range(config.NUM_CLASSES):
        title = _get_legend_title_from_list(config.CLASS_NAMES, class_index)
        try:
            roc = _get_roc(y_trues, y_preds, class_index, label_type)
            color = _get_colors(config.NUM_CLASSES)[class_index]
            robjects.r.plot(
                roc,
                add=(class_index != 0),
                col=color,
            )
        except RRuntimeError:
            traceback.print_exc()
            print(f'Plotting {title} in the all-in-one ROC failed.')
        else:
            legend = class_roc_metrics_list[class_index].legend
            legends.append(legend)
            colors.append(color)

    try:
        robjects.r.legend(
            'bottomright',
            legend=robjects.StrVector(legends),
            fill=robjects.StrVector(colors),
            cex=0.75,
        )
        print(f'The all-in-one ROC has been saved as {roc_path} .')
    except RRuntimeError:
        traceback.print_exc()
        print('Plotting the all-in-one ROC failed.')

    # Plot indivisual ROCs if required
    for class_index in range(config.NUM_CLASSES):
        title = _get_legend_title_from_list(config.CLASS_NAMES, class_index)
        try:
            r_package_grdevices = _install_and_load_r_package('grDevices')
            roc_path = str(args.roc_output / f'roc_class{class_index}.pdf')
            r_package_grdevices.pdf(file=roc_path)

            roc = _get_roc(y_trues, y_preds, class_index, label_type)
            color = _get_colors(config.NUM_CLASSES)[class_index]
            robjects.r.plot(
                roc,
                col=color,
            )

            legend = class_roc_metrics_list[class_index].legend
            robjects.r.legend(
                'bottomright',
                legend=robjects.StrVector([legend]),
                fill=robjects.StrVector([color]),
                cex=0.75,
            )

            print(f'The ROC of {title} has been saved as {roc_path} .')

        except RRuntimeError:
            traceback.print_exc()
            print(f'Plotting the ROC of {title} failed.')


def _install_and_load_r_package(
    name: str,
) -> t.Union[rpackages.InstalledSTPackage, rpackages.InstalledPackage]:
    try:
        package = rpackages.importr(name)
    except rpackages.PackageNotInstalledError:
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(robjects.StrVector([name]))
        package = rpackages.importr(name)

    return package


def _get_roc(
    y_trues: np.ndarray,
    y_preds: np.ndarray,
    class_index: int,
    label_type: str,
) -> robjects.ListVector:
    r_package_proc = _install_and_load_r_package('pROC')

    # Get the ground truths
    if label_type == 'multi_class':
        y_trues_of_class = (y_trues == class_index).astype(float)
    elif label_type == 'multi_label':
        y_trues_of_class = y_trues[:, class_index]
    else:
        raise ValueError(f'Unknown label_type: {label_type}')

    # Remove don't care
    is_dont_care = np.isnan(y_trues_of_class)
    y_trues_of_class = y_trues_of_class[~is_dont_care]
    y_preds_of_class = y_preds[:, class_index][~is_dont_care]

    # Get the ROC
    response = robjects.FloatVector(y_trues_of_class)
    predictor = robjects.FloatVector(y_preds_of_class)
    roc = r_package_proc.roc(
        response=response,
        predictor=predictor,
        quiet=True,
        auc=True,
        ci=True,
    )
    return roc


def _test_roc_with_null_hypothesis(roc: robjects.ListVector) -> robjects.ListVector:
    r_package_proc = _install_and_load_r_package('pROC')

    response = roc[roc.names.index('response')]
    null_roc = r_package_proc.roc(
        response=response,
        predictor=robjects.FloatVector([0.0 for _ in response]),
        quiet=True,
    )
    null_hypothesis_test_result = r_package_proc.roc_test(
        roc1=roc,
        roc2=null_roc,
        method='delong',
        alternative='two.sided',
        paired=True,
    )
    return null_hypothesis_test_result


def _get_colors(num_colors: int) -> t.List[str]:
    color_map = matplotlib.cm.Set1
    if num_colors > color_map.N:
        raise ValueError('Too many colors to draw.')
    colors = [matplotlib.colors.rgb2hex(color_map(idx)) for idx in range(num_colors)]
    return colors


def _get_legend_title_from_list(titles: t.Optional[t.List[str]], index: int) -> str:
    if titles is None:
        warnings.warn('Please set `COLOR_NAMES` in the config for better expression.')
        title = f'class {index}'
    else:
        if index >= len(titles):
            raise ValueError('`titles` contains too few legend titles.')
        title = titles[index]

    return title


def _get_roc_curve(roc: robjects.ListVector) -> t.List[t.Tuple[float, float]]:
    false_positive_rates = 1.0 - np.array(roc[roc.names.index('specificities')])
    true_positive_rates = np.array(roc[roc.names.index('sensitivities')])
    return list(zip(false_positive_rates, true_positive_rates))


if __name__ == '__main__':
    main()
