

import numpy as np


R = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]])


def homogeneous_transform(θ1, θ2, θ3, l1, l2, l3, i: int, should_substitute=True) -> np.ndarray:
    """
    Returns the HT for the i-th phalange, located at the end of the i-th phalange.
    
    :param i: I'th phalange (1, 2, or 3)
    :type i: int

    :param should_substitute: Whether to substitute the total angles and their derivatives
    :type should_substitute: bool
    """
    θ_tot = sum([θ1, θ2, θ3][:i])  # Total angle up to the i-th phalange
    d = sum(
        (
            R(sum([θ1, θ2, θ3][:j]))
            @ np.array([[[l1, l2, l3][j - 1]], [0]])
            for j in range(1, i + 1)
        ),
        np.zeros((2, 1)),
    )
    HT = np.block([
        [R(θ_tot), d],
        [np.zeros((1, 2)), 1]
    ])
    return HT
