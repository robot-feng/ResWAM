import numpy as np
import pytest

from examples.modelExtensions.MiniCPM.libero_dual_pilot_eval import (
    PILOT_EXECUTION_HORIZON,
    _select_executed_action,
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
