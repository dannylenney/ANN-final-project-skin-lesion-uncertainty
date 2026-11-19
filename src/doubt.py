"""Per-image doubt scores. Members must use the same image and class order."""

import numpy as np


def softmax_doubt(probs):
    """Return one minus the top probability for each row of (images, classes)."""
    return 1 - np.max(probs, axis=1)


def ensemble_entropy(probs):
    """Return entropy in nats after averaging (seeds, images, classes) over seeds."""
    mean_probs = np.mean(probs, axis=0, dtype=float)
    log_probs = np.log(mean_probs, where=mean_probs > 0, out=np.zeros_like(mean_probs))
    return -np.sum(mean_probs * log_probs, axis=1)


def _vote_disagreement(probs):
    """Return the fraction outside the plurality. Each vote uses the first maximum."""
    votes = np.argmax(probs, axis=2)
    counts = [(votes == label).sum(axis=0) for label in range(np.shape(probs)[2])]
    return 1 - np.max(counts, axis=0) / len(votes)


def seed_disagreement(probs):
    """Score disagreement across seeds given (seeds, images, classes)."""
    return _vote_disagreement(probs)


def cross_model_disagreement(probs):
    """Score (models, images, classes) with each model already averaged over seeds."""
    return _vote_disagreement(probs)
