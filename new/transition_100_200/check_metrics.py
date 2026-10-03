"""Small CPU fixtures for the transition_100_200 summary contract."""
from __future__ import annotations

import json
import unittest

import numpy as np

from metrics import summarize


def _inputs(horizon: int, maps: int, height: int, width: int):
    trace = np.zeros((horizon + 1, maps, height, width), dtype=bool)
    changed = np.ones((maps, height, width), dtype=bool)
    distance = np.full((maps, height, width), 20, dtype=np.int16)
    return trace, changed, distance


class TransitionMetricsChecks(unittest.TestCase):
    def test_transient_wrong_then_recover_has_no_endpoint_d_but_first_exit(self):
        trace = np.ones((129, 1, 1, 1), dtype=bool)
        trace[70, 0, 0, 0] = False
        result = summarize(trace, np.ones((1, 1, 1), dtype=bool), np.full((1, 1, 1), 20))
        json.dumps(result, allow_nan=False)
        interval = result["intervals"]["64_128"]
        self.assertEqual(interval["endpoint_destruction"]["pooled_numerator"], 0)
        self.assertEqual(interval["first_exit"]["pooled_numerator"], 1)
        self.assertEqual(interval["continuous_survival"]["pooled_numerator"], 0)
        self.assertEqual(interval["first_exit"]["pooled_denominator"], 1)
        finite_lag = result["finite_horizon_lags"]["primary"]["lag_final_correct"]
        self.assertEqual(finite_lag["pooled_median"], 71.0)

    def test_late_first_correct_is_right_censored_at_lag_eight(self):
        trace, changed, distance = _inputs(20, 1, 1, 1)
        trace[18:, 0, 0, 0] = True
        result = summarize(trace, changed, distance)
        lag8 = result["fixed_lag_first_correct_survival"]["primary"]["lags"][0]
        self.assertEqual(lag8["uninterrupted_survival"]["pooled_denominator"], 0)
        self.assertIsNone(lag8["uninterrupted_survival"]["pooled_rate"])
        self.assertEqual(lag8["exclusions"]["never_correct_pooled"], 0)
        self.assertEqual(lag8["exclusions"]["right_censored_pooled"], 1)

    def test_never_correct_and_final_wrong_have_undefined_stable_lag(self):
        trace, changed, distance = _inputs(4, 1, 1, 2)
        trace[1:3, 0, 0, 1] = True
        finite = summarize(trace, changed, distance)["finite_horizon_lags"]["primary"]["pooled"]
        self.assertEqual(finite["never_correct"], 1)
        self.assertEqual(finite["final_wrong"], 2)
        self.assertEqual(finite["final_correct"], 0)
        self.assertEqual(finite["ineligible_stable_time_final_wrong"], 2)
        self.assertIsNone(finite["stable_time_median_final_correct"])
        self.assertIsNone(finite["first_correct_median_final_correct"])

    def test_current_run_age_resets_after_regression(self):
        trace = np.asarray([True, False, True, True, False], dtype=bool).reshape(5, 1, 1, 1)
        result = summarize(trace, np.ones((1, 1, 1), dtype=bool), np.full((1, 1, 1), 20))
        hazard = result["age_hazard"]["by_distance"]["17_31"]
        self.assertEqual(hazard["1"]["pooled_numerator"], 1)
        self.assertEqual(hazard["1"]["pooled_denominator"], 2)
        self.assertEqual(hazard["2_4"]["pooled_numerator"], 1)
        self.assertEqual(hazard["2_4"]["pooled_denominator"], 1)

    def test_weighted_net_gain_and_negative_distance_wall_sentinel(self):
        trace, changed, distance = _inputs(128, 2, 1, 3)
        changed[0, 0, 1:] = False
        distance[0, 0, 1:] = -1  # Unselected wall sentinel is allowed.
        trace[64, 1, 0, :] = True
        trace[65:, 1, 0, 1:] = True
        trace[128, 0, 0, 0] = True
        net = summarize(trace, changed, distance)["intervals"]["64_128"]["net_gain"]
        self.assertAlmostEqual(net["equal_map_mean"], 1.0 / 3.0)
        self.assertEqual(net["pooled_numerator"], 0)
        self.assertEqual(net["pooled_denominator"], 4)
        self.assertEqual(net["pooled_rate"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
