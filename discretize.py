import numpy as np


def discretize_state(state, bins=(8, 8, 8)):
    """
    Discretize continuous state into integer bins.
    State: [velocity, horizontal_dist, vertical_dist]
    """
    low = np.array([-1.5, 0.0, -1.5])
    high = np.array([1.5, 3.0, 1.5])

    state_clipped = np.clip(state, low, high)
    ratios = (state_clipped - low) / (high - low)
    ratios = np.clip(ratios, 0.0, 0.999)

    idx = tuple((ratios * np.array(bins)).astype(int))
    return idx
