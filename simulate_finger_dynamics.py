import numpy as np


def mass_matrix(q1, q2, q3, m1, m2, m3, l1, l2, l3, r1, r2, r3):
    """
    3x3 mass matrix M(q) for a 3-link planar finger.
    All links are modelled as solid cylinders.

    Parameters
    ----------
    q1, q2, q3   : joint angles [rad]
    m1, m2, m3   : link masses  [kg]
    l1, l2, l3   : link lengths [m]
    r1, r2, r3   : link radii   [m]

    Returns
    -------
    M : np.ndarray, shape (3, 3)
    """
    M = np.array([
        [(7/12)*l1**2*m1 + l1**2*m2 + l1**2*m3 + l1*l2*m2*np.cos(q2) + 2*l1*l2*m3*np.cos(q2) + l1*l3*m3*np.cos(q2 + q3) + (1/4)*l2**2*m2 + l2**2*m3 + l2*l3*m3*np.cos(q3) + (1/4)*l3**2*m3 + (1/4)*m1*r1**2, (1/2)*l1*l2*m2*np.cos(q2) + l1*l2*m3*np.cos(q2) + (1/2)*l1*l3*m3*np.cos(q2 + q3) + (1/4)*l2**2*m2 + l2**2*m3 + l2*l3*m3*np.cos(q3) + (1/4)*l3**2*m3, (1/4)*l3*m3*(2*l1*np.cos(q2 + q3) + 2*l2*np.cos(q3) + l3)],
        [(1/2)*l1*l2*m2*np.cos(q2) + l1*l2*m3*np.cos(q2) + (1/2)*l1*l3*m3*np.cos(q2 + q3) + (1/4)*l2**2*m2 + l2**2*m3 + l2*l3*m3*np.cos(q3) + (1/4)*l3**2*m3, (7/12)*l2**2*m2 + l2**2*m3 + l2*l3*m3*np.cos(q3) + (1/4)*l3**2*m3 + (1/4)*m2*r2**2, (1/4)*l3*m3*(2*l2*np.cos(q3) + l3)],
        [(1/4)*l3*m3*(2*l1*np.cos(q2 + q3) + 2*l2*np.cos(q3) + l3), (1/4)*l3*m3*(2*l2*np.cos(q3) + l3), (1/12)*m3*(7*l3**2 + 3*r3**2)]
    ])
    return M