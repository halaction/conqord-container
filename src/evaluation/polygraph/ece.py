# implementaion from https://github.com/TaoShuchang/CONQORD/tree/master

import numpy as np
from typing import List

from lm_polygraph.ue_metrics.ue_metric import UEMetric, normalize

class ExpectedCalibrationError(UEMetric):
    def __init__(self, n_bins: float = 10):
        super().__init__()
        self.n_bins = n_bins

    def __str__(self):
        return "ece"

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        """
        Parameters:
            estimator (List[int]): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (List[int]): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: ECE uncertainty estimations.
                Lower values indicate better uncertainty estimations.
        """
        estimator = 1 - np.array(normalize(estimator))
        target = np.array(normalize(target))
        bin_bounds = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_bounds[:-1], bin_bounds[1:]):
            in_bin = np.where((estimator > bin_lower) & (estimator <= bin_upper))[0]
            if len(in_bin) > 0:
                bin_accuracy = np.mean(target[in_bin])
                bin_confidence = np.mean(estimator[in_bin])
                bin_proportion = len(in_bin) / float(len(estimator))
                ece += np.abs(bin_accuracy - bin_confidence) * bin_proportion

        return ece
