
import numpy as np
from numpy import sin, cos, pi

# Parameters for the finger dynamics
m1, m2, m3 = 0.02, 0.015, 20  # Masses [kg]
l1, l2, l3 = 0.048, 0.030, 0.024  # Lengths [m]
r1, r2, r3 = 0.085, 0.085, 0.085  # Radii [m]

k1, k2, k3 = 10.0, 10.0, 10.0  # Spring constants [N*m/rad]
b1, b2, b3 = 2.0, 2.0, 2.0  # Damping coefficients [N*m*s/rad]

theta1_0, theta2_0, theta3_0 = pi/6, pi/4, pi/12  # Spring rest angles [rad]


def Tau_K(theta: np.ndarray) -> np.ndarray:
    """
    3x1 vector of spring forces.
    
    Parameters
    ----------
    theta : np.ndarray, shape (3,) Relative joint angles [rad]
    theta1_0, theta2_0, theta3_0 : spring rest angles [rad]
    k1, k2, k3   : spring constants [N*m/rad]

    Returns
    -------
    Tau_K : np.ndarray, shape (3, 1)
    """
    theta1, theta2, theta3 = theta
    Tau_K = np.array([
        k1*(theta1 - theta1_0),
        k2*(theta2 - theta2_0),
        k3*(theta3 - theta3_0),
    ])
    return Tau_K


def Tau_B(theta_dot: np.ndarray) -> np.ndarray:
    """
    3x1 vector of damping forces.
    
    Parameters
    ----------
    theta_dot    : np.ndarray, shape (3,) Joint angular velocities [rad/s]
    b1, b2, b3   : damping coefficients [N*m*s/rad]

    Returns
    -------
    Tau_B : np.ndarray, shape (3, 1)
    """
    theta1_dot, theta2_dot, theta3_dot = theta_dot
    Tau_B = np.array([
        b1*theta1_dot,
        b2*theta2_dot,
        b3*theta3_dot,
    ])
    return Tau_B


def C(theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
    """
    3x3 Coriolis and centrifugal matrix C(q, q_dot) for a 3-link planar finger.
    All links are modelled as solid cylinders.
    
    Parameters
    ----------
    theta : np.ndarray, shape (3,) Joint angles [rad]
    theta_dot : np.ndarray, shape (3,) Joint angular velocities [rad/s]
    m1, m2, m3   : link masses  [kg]
    l1, l2, l3   : link lengths [m]
    r1, r2, r3   : link radii   [m]
    
    Returns
    -------
    C : np.ndarray, shape (3, 3)
    """
    theta1, theta2, theta3 = theta
    theta1_dot, theta2_dot, theta3_dot = theta_dot
    return np.array([
        [-l1*theta2_dot*(l2*m2*sin(theta2) + 2*l2*m3*sin(theta2) + l3*m3*sin(theta2 + theta3))/2 - l3*m3*theta3_dot*(l1*sin(theta2 + theta3) + l2*sin(theta3))/2, -l1*theta1_dot*(l2*m2*sin(theta2) + 2*l2*m3*sin(theta2) + l3*m3*sin(theta2 + theta3))/2 - l1*theta2_dot*(l2*m2*sin(theta2) + 2*l2*m3*sin(theta2) + l3*m3*sin(theta2 + theta3))/2 - l3*m3*theta3_dot*(l1*sin(theta2 + theta3) + l2*sin(theta3))/2, l3*m3*(l1*sin(theta2 + theta3) + l2*sin(theta3))*(-theta1_dot - theta2_dot - theta3_dot)/2],
        [l1*theta1_dot*(l2*m2*sin(theta2) + 2*l2*m3*sin(theta2) + l3*m3*sin(theta2 + theta3))/2 - l2*l3*m3*theta3_dot*sin(theta3)/2, -l2*l3*m3*theta3_dot*sin(theta3)/2, l2*l3*m3*(-theta1_dot - theta2_dot - theta3_dot)*sin(theta3)/2],
        [l3*m3*(l2*theta2_dot*sin(theta3) + theta1_dot*(l1*sin(theta2 + theta3) + l2*sin(theta3)))/2, l2*l3*m3*(theta1_dot + theta2_dot)*sin(theta3)/2, 0],
    ])


def M(theta: np.ndarray) -> np.ndarray:
    """
    3x3 mass matrix M(q) for a 3-link planar finger.
    All links are modelled as solid cylinders.

    Parameters
    ----------
    theta : np.ndarray, shape (3,) Joint angles [rad]
    m1, m2, m3   : link masses  [kg]
    l1, l2, l3   : link lengths [m]
    r1, r2, r3   : link radii   [m]

    Returns
    -------
    M : np.ndarray, shape (3, 3)
    """
    theta1, theta2, theta3 = theta
    return np.array([
        [7*l1**2*m1/12 + l1**2*m2 + l1**2*m3 + l1*l2*m2*cos(theta2) + 2*l1*l2*m3*cos(theta2) + l1*l3*m3*cos(theta2 + theta3) + 7*l2**2*m2/12 + l2**2*m3 + l2*l3*m3*cos(theta3) + 7*l3**2*m3/12 + m1*r1**2/4 + m2*r2**2/4 + m3*r3**2/4, l1*l2*m2*cos(theta2)/2 + l1*l2*m3*cos(theta2) + l1*l3*m3*cos(theta2 + theta3)/2 + 7*l2**2*m2/12 + l2**2*m3 + l2*l3*m3*cos(theta3) + 7*l3**2*m3/12 + m2*r2**2/4 + m3*r3**2/4, m3*(6*l1*l3*cos(theta2 + theta3) + 6*l2*l3*cos(theta3) + 7*l3**2 + 3*r3**2)/12],
        [l1*l2*m2*cos(theta2)/2 + l1*l2*m3*cos(theta2) + l1*l3*m3*cos(theta2 + theta3)/2 + 7*l2**2*m2/12 + l2**2*m3 + l2*l3*m3*cos(theta3) + 7*l3**2*m3/12 + m2*r2**2/4 + m3*r3**2/4, 7*l2**2*m2/12 + l2**2*m3 + l2*l3*m3*cos(theta3) + 7*l3**2*m3/12 + m2*r2**2/4 + m3*r3**2/4, m3*(6*l2*l3*cos(theta3) + 7*l3**2 + 3*r3**2)/12],
        [m3*(6*l1*l3*cos(theta2 + theta3) + 6*l2*l3*cos(theta3) + 7*l3**2 + 3*r3**2)/12, m3*(6*l2*l3*cos(theta3) + 7*l3**2 + 3*r3**2)/12, m3*(7*l3**2 + 3*r3**2)/12],
    ])

def dynamics(theta: np.ndarray, theta_dot: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    Compute the joint accelerations given the current joint angles and velocities.
    
    Parameters
    ----------
    theta : np.ndarray, shape (3,)
        Current joint angles [rad]
    theta_dot : np.ndarray, shape (3,)
        Current joint angular velocities [rad/s]
    tau : np.ndarray, shape (3,)
    Returns
    -------
    theta_ddot : np.ndarray, shape (3,)
        Joint angular accelerations [rad/s^2]
    """
    Tau_K_val = Tau_K(theta)
    Tau_B_val = Tau_B(theta_dot)
    C_val = C(theta, theta_dot)
    M_val = M(theta)
    
    # Compute the joint accelerations using M*theta_ddot = Tau_net
    theta_ddot = np.linalg.solve(M_val, -Tau_K_val - Tau_B_val - C_val @ theta_dot + tau)
    
    return theta_ddot


