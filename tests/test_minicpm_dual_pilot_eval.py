import numpy as np
import pytest

from examples.modelExtensions.MiniCPM.libero_dual_pilot_eval import (
    PILOT_EXECUTION_HORIZON,
    _select_executed_action,
    _unnormalize_action,
)


def test_pilot_executes_one_action_from_predicted_action_chunk():
    predicted_chunk = np.arange(8 * 7, dtype=np.float32).reshape(8, 7)

    selected = _select_executed_action(predicted_chunk)

    assert PILOT_EXECUTION_HORIZON == 1
    np.testing.assert_array_equal(selected, predicted_chunk[0])


def test_pilot_accepts_single_action_and_rejects_empty_chunk():
    single_action = np.arange(7, dtype=np.float32)
    np.testing.assert_array_equal(_select_executed_action(single_action), single_action)
    with pytest.raises(ValueError, match="predicted 0 actions"):
        _select_executed_action(np.empty((0, 7), dtype=np.float32))


def test_unnormalization_maps_eef_extremes_and_gripper_convention():
    stats = {"min": [-2.0, -3.0, -4.0, -0.5, -0.6, -0.7], "max": [2.0, 3.0, 4.0, 0.5, 0.6, 0.7]}

    closed = _unnormalize_action(np.array([-1, -1, -1, -1, -1, -1, 1], dtype=np.float32), stats)
    opened = _unnormalize_action(np.array([1, 1, 1, 1, 1, 1, 0], dtype=np.float32), stats)

    np.testing.assert_allclose(closed[:6], stats["min"])
    np.testing.assert_allclose(opened[:6], stats["max"])
    assert closed[6] == -1.0
    assert opened[6] == 1.0
