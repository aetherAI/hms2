from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pydantic


class BinaryMetrics(pydantic.BaseModel):
    recall: float = np.nan
    precision: float = np.nan
    specificity: float = np.nan
    npv: float = np.nan
    mcc: float = np.nan
    f1_score: float = np.nan

    @staticmethod
    def from_predictions(
        y_true: npt.NDArray[np.bool_],
        y_pred: npt.NDArray[np.bool_],
    ) -> BinaryMetrics:
        tp = (y_true & y_pred).astype(int).sum()
        fn = (y_true & ~y_pred).astype(int).sum()
        fp = (~y_true & y_pred).astype(int).sum()
        tn = (~y_true & ~y_pred).astype(int).sum()

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        specificity = tn / (tn + fp)
        npv = tn / (tn + fn)
        mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        f1_score = (2.0 * tp) / (2.0 * tp + fn + fp)

        return BinaryMetrics(
            recall=recall,
            precision=precision,
            specificity=specificity,
            npv=npv,
            mcc=mcc,
            f1_score=f1_score,
        )
