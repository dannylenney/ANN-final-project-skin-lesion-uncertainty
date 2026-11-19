import numpy as np
import pytest

from src.doubt import (
    cross_model_disagreement,
    ensemble_entropy,
    seed_disagreement,
    softmax_doubt,
)


def test_softmax_doubt():
    probs = [[0.2, 0.8], [1, 0]]
    np.testing.assert_allclose(softmax_doubt(probs), [0.2, 0])


def test_entropy_and_votes():
    probs = [
        [[1, 0], [0.51, 0.49], [0, 1]],
        [[0, 1], [0.51, 0.49], [0, 1]],
    ]
    np.testing.assert_allclose(ensemble_entropy(probs), [np.log(2), 0.6929471672, 0])
    np.testing.assert_allclose(seed_disagreement(probs), [0.5, 0, 0])


@pytest.mark.parametrize("score", [seed_disagreement, cross_model_disagreement])
def test_disagreement(score):
    probs = np.array([
        [[0.5, 0.5], [1, 0]],
        [[1, 0], [0, 1]],
        [[1, 0], [1, 0]],
        [[1, 0], [0, 1]],
    ])
    np.testing.assert_allclose(score(probs[:3]), [0, 1 / 3])
    np.testing.assert_allclose(score(probs), [0, 0.5])


def test_three_distinct_class_votes():
    probs = np.eye(3)[:, None, :]
    np.testing.assert_allclose(ensemble_entropy(probs), [np.log(3)])
    np.testing.assert_allclose(seed_disagreement(probs), [2 / 3])
    np.testing.assert_allclose(cross_model_disagreement(probs), [2 / 3])
