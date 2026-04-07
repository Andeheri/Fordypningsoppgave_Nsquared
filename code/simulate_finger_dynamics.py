import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from visualization_finger_simulation import animate_finger_simulation, plot_simulation_angles

"""
----------------------- Define Constants -----------------------
"""
m1, m2, m3 = 0.02, 0.015, 20  # Masses [kg]
l1, l2, l3 = 0.048, 0.030, 0.024  # Lengths [m]
r1, r2, r3 = 0.085, 0.085, 0.085  # Radii [m]
N_max = 22 # Maybe 30
cable_speed = 67  # mm/s
k1, k2, k3 = 10.0, 10.0, 10.0  # Spring constants [N*m/rad]
b1, b2, b3 = 2.0, 2.0, 2.0  # Damping coefficients [N*m*s/rad]

theta1_0, theta2_0, theta3_0 = pi/6, pi/4, pi/12  # Spring rest angles [rad]

# ---- Three link forces ----------------------------------------
# Attachment positions: distance from the proximal joint of each link [m]
# Allowed ranges:  0 <= force_s1 <= l1,  0 <= force_s2 <= l2,  0 <= force_s3 <= l3
force_s1 = l1 * 0.5   # link 1 – applied at midpoint
force_s2 = l2 * 0.5   # link 2 – applied at midpoint
force_s3 = l3 * 0.5   # link 3 – applied at midpoint

# Force magnitude at each timestep [N] — callables  F(t) -> float.
# Set  lambda t: 0.0  to disable a force.
F_link1 = lambda t: 25.0
F_link2 = lambda t: 25.0
F_link3 = lambda t: 25.0

# Relative force angle at each timestep [rad] — callables  alpha(t) -> float.
#   alpha = 0      → force is directed along the link axis (proximal → distal)
#   alpha = pi/2   → force is 90° counter-clockwise from the link axis
alpha_link1 = lambda t: pi/4
alpha_link2 = lambda t: pi/2
alpha_link3 = lambda t: pi / 2

should_apply_link_forces = True

# Initial conditions: all angles and velocities zero
state0 = [0.0, 0.0, 0.0,  # theta1, theta2, theta3
          0.0, 0.0, 0.0]  # theta1_dot, theta2_dot, theta3_dot
T = 1.0  # Total simulation time [s]
simulation_speed = 1.0  # Playback speed multiplier (0.5 = half speed, 1.0 = real-time, etc.)

should_save_animation = True  # Set to True to save the animation as a GIF file
note = "test3"  # A note to include in the filename for clarity
save_folder = "figures"

filename_angles    = f"{save_folder}/finger_simulation_angles_{note}.png"  # Filename for the saved angles plot
filename_animation = f"{save_folder}/finger_simulation_{note}.gif"         # Filename for the saved animation

"""
----------------------- Simulation and Visualization -----------------------
"""
def Tau_K(theta1, theta2, theta3):
    """
    3x1 vector of spring forces.
    
    Parameters
    ----------
    theta1, theta2, theta3   : Relative joint angles [rad]
    theta1_0, theta2_0, theta3_0 : spring rest angles [rad]
    k1, k2, k3   : spring constants [N*m/rad]

    Returns
    -------
    Tau_K : np.ndarray, shape (3, 1)
    """
    Tau_K = np.array([
        k1*(theta1 - theta1_0),
        k2*(theta2 - theta2_0),
        k3*(theta3 - theta3_0),
    ])
    return Tau_K


def Tau_B(theta1_dot, theta2_dot, theta3_dot):
    """
    3x1 vector of damping forces.
    
    Parameters
    ----------
    theta1_dot, theta2_dot, theta3_dot : joint angular velocities [rad/s]
    b1, b2, b3   : damping coefficients [N*m*s/rad]

    Returns
    -------
    Tau_B : np.ndarray, shape (3, 1)
    """
    Tau_B = np.array([
        b1*theta1_dot,
        b2*theta2_dot,
        b3*theta3_dot,
    ])
    return Tau_B


def C(theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot):
    """
    3x3 Coriolis and centrifugal matrix C(q, q_dot) for a 3-link planar finger.
    All links are modelled as solid cylinders.
    
    Parameters
    ----------
    theta1, theta2, theta3   : joint angles [rad]
    theta1_dot, theta2_dot, theta3_dot : joint angular velocities [rad/s]
    m1, m2, m3   : link masses  [kg]
    l1, l2, l3   : link lengths [m]
    r1, r2, r3   : link radii   [m]
    
    Returns
    -------
    C : np.ndarray, shape (3, 3)
    """
    return np.array([
        [-l1*theta2_dot*(l2*m2*sin(theta2) + 2*l2*m3*sin(theta2) + l3*m3*sin(theta2 + theta3))/2 - l3*m3*theta3_dot*(l1*sin(theta2 + theta3) + l2*sin(theta3))/2, -l1*theta1_dot*(l2*m2*sin(theta2) + 2*l2*m3*sin(theta2) + l3*m3*sin(theta2 + theta3))/2 - l1*theta2_dot*(l2*m2*sin(theta2) + 2*l2*m3*sin(theta2) + l3*m3*sin(theta2 + theta3))/2 - l3*m3*theta3_dot*(l1*sin(theta2 + theta3) + l2*sin(theta3))/2, l3*m3*(l1*sin(theta2 + theta3) + l2*sin(theta3))*(-theta1_dot - theta2_dot - theta3_dot)/2],
        [l1*theta1_dot*(l2*m2*sin(theta2) + 2*l2*m3*sin(theta2) + l3*m3*sin(theta2 + theta3))/2 - l2*l3*m3*theta3_dot*sin(theta3)/2, -l2*l3*m3*theta3_dot*sin(theta3)/2, l2*l3*m3*(-theta1_dot - theta2_dot - theta3_dot)*sin(theta3)/2],
        [l3*m3*(l2*theta2_dot*sin(theta3) + theta1_dot*(l1*sin(theta2 + theta3) + l2*sin(theta3)))/2, l2*l3*m3*(theta1_dot + theta2_dot)*sin(theta3)/2, 0],
    ])


def M(theta1, theta2, theta3):
    """
    3x3 mass matrix M(q) for a 3-link planar finger.
    All links are modelled as solid cylinders.

    Parameters
    ----------
    theta1, theta2, theta3   : joint angles [rad]
    m1, m2, m3   : link masses  [kg]
    l1, l2, l3   : link lengths [m]
    r1, r2, r3   : link radii   [m]

    Returns
    -------
    M : np.ndarray, shape (3, 3)
    """
    return np.array([
        [7*l1**2*m1/12 + l1**2*m2 + l1**2*m3 + l1*l2*m2*cos(theta2) + 2*l1*l2*m3*cos(theta2) + l1*l3*m3*cos(theta2 + theta3) + 7*l2**2*m2/12 + l2**2*m3 + l2*l3*m3*cos(theta3) + 7*l3**2*m3/12 + m1*r1**2/4 + m2*r2**2/4 + m3*r3**2/4, l1*l2*m2*cos(theta2)/2 + l1*l2*m3*cos(theta2) + l1*l3*m3*cos(theta2 + theta3)/2 + 7*l2**2*m2/12 + l2**2*m3 + l2*l3*m3*cos(theta3) + 7*l3**2*m3/12 + m2*r2**2/4 + m3*r3**2/4, m3*(6*l1*l3*cos(theta2 + theta3) + 6*l2*l3*cos(theta3) + 7*l3**2 + 3*r3**2)/12],
        [l1*l2*m2*cos(theta2)/2 + l1*l2*m3*cos(theta2) + l1*l3*m3*cos(theta2 + theta3)/2 + 7*l2**2*m2/12 + l2**2*m3 + l2*l3*m3*cos(theta3) + 7*l3**2*m3/12 + m2*r2**2/4 + m3*r3**2/4, 7*l2**2*m2/12 + l2**2*m3 + l2*l3*m3*cos(theta3) + 7*l3**2*m3/12 + m2*r2**2/4 + m3*r3**2/4, m3*(6*l2*l3*cos(theta3) + 7*l3**2 + 3*r3**2)/12],
        [m3*(6*l1*l3*cos(theta2 + theta3) + 6*l2*l3*cos(theta3) + 7*l3**2 + 3*r3**2)/12, m3*(6*l2*l3*cos(theta3) + 7*l3**2 + 3*r3**2)/12, m3*(7*l3**2 + 3*r3**2)/12],
    ])


def tau_link_forces(theta1, theta2, theta3, t):
    """
    Generalized torques from three link forces, via the virtual-work principle.

    A separate force acts on each link, applied at a fixed distance along that
    link (measured from its proximal joint) and directed at a time-varying angle
    relative to the link's own axis in the global frame:

        alpha = 0      → force is directed along the link axis (proximal → distal)
        alpha = pi/2   → force is 90° CCW from the link axis

    The world-frame force vector for link i is

        F_i = F_link_i(t) * [cos(a_i + alpha_i(t)),
                             sin(a_i + alpha_i(t))]

    where the cumulative joint angles are

        a1 = theta1,   a2 = theta1+theta2,   a3 = theta1+theta2+theta3.

    Attachment-point world coordinates
    -----------------------------------
        p_att1 = force_s1*[cos(a1), sin(a1)]
        p_att2 = l1*[cos(a1), sin(a1)]  +  force_s2*[cos(a2), sin(a2)]
        p_att3 = l1*[cos(a1), sin(a1)]  +  l2*[cos(a2), sin(a2)]  +  force_s3*[cos(a3), sin(a3)]

    Positional Jacobians  J_i  (2x3)  of each attachment point
    -----------------------------------------------------------
        J1 = [[ -force_s1*sin(a1),                                0,                       0 ],
              [  force_s1*cos(a1),                                0,                       0 ]]

        J2 = [[ -l1*sin(a1) - force_s2*sin(a2),  -force_s2*sin(a2),                       0 ],
              [  l1*cos(a1) + force_s2*cos(a2),   force_s2*cos(a2),                       0 ]]

        J3 = [[ -l1*sin(a1) - l2*sin(a2) - force_s3*sin(a3),  -l2*sin(a2) - force_s3*sin(a3),  -force_s3*sin(a3) ],
              [  l1*cos(a1) + l2*cos(a2) + force_s3*cos(a3),   l2*cos(a2) + force_s3*cos(a3),   force_s3*cos(a3) ]]

    Generalized torques (virtual-work principle)
    -------------------------------------------
        tau_i     = J_i^T @ F_i
        tau_total = tau_1 + tau_2 + tau_3

    Parameters
    ----------
    theta1, theta2, theta3 : joint angles [rad]
    t                      : current simulation time [s]

    Returns
    -------
    tau : np.ndarray, shape (3,)   generalized torques [N*m]
    """
    a1 = theta1
    a2 = theta1 + theta2
    a3 = theta1 + theta2 + theta3

    # ---- Link 1 force ----
    # Attachment: p_att1 = force_s1 * [cos(a1), sin(a1)]
    J1 = np.array([
        [-force_s1 * sin(a1),  0.0,  0.0],
        [ force_s1 * cos(a1),  0.0,  0.0],
    ])
    F1 = F_link1(t) * np.array([cos(a1 + alpha_link1(t)), sin(a1 + alpha_link1(t))])
    tau1 = J1.T @ F1

    # ---- Link 2 force ----
    # Attachment: p_att2 = l1*[cos(a1), sin(a1)] + force_s2*[cos(a2), sin(a2)]
    J2 = np.array([
        [-l1*sin(a1) - force_s2*sin(a2),  -force_s2*sin(a2),  0.0],
        [ l1*cos(a1) + force_s2*cos(a2),   force_s2*cos(a2),  0.0],
    ])
    F2 = F_link2(t) * np.array([cos(a2 + alpha_link2(t)), sin(a2 + alpha_link2(t))])
    tau2 = J2.T @ F2

    # ---- Link 3 force ----
    # Attachment: p_att3 = l1*[cos(a1),sin(a1)] + l2*[cos(a2),sin(a2)] + force_s3*[cos(a3),sin(a3)]
    J3 = np.array([
        [-l1*sin(a1) - l2*sin(a2) - force_s3*sin(a3),  -l2*sin(a2) - force_s3*sin(a3),  -force_s3*sin(a3)],
        [ l1*cos(a1) + l2*cos(a2) + force_s3*cos(a3),   l2*cos(a2) + force_s3*cos(a3),   force_s3*cos(a3)],
    ])
    F3 = F_link3(t) * np.array([cos(a3 + alpha_link3(t)), sin(a3 + alpha_link3(t))])
    tau3 = J3.T @ F3

    return tau1 + tau2 + tau3



def dynamics(t, state):
    """
    State vector: [theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot]
    EOM: M(q)*q_ddot + C(q,q_dot)*q_dot + Tau_K(q) + Tau_B(q_dot) = 0
    => q_ddot = M^{-1} * (-C*q_dot - Tau_K - Tau_B)
    """
    th1, th2, th3, th1d, th2d, th3d = state
    q_dot = np.array([th1d, th2d, th3d])

    M_mat = M(th1, th2, th3)
    C_mat = C(th1, th2, th3, th1d, th2d, th3d)
    tau_k = Tau_K(th1, th2, th3)
    tau_b = Tau_B(th1d, th2d, th3d)
    tau_ext = tau_link_forces(th1, th2, th3, t) if should_apply_link_forces else np.zeros(3)

    rhs = -C_mat @ q_dot - tau_k - tau_b + tau_ext
    q_ddot = np.linalg.solve(M_mat, rhs)
    return [th1d, th2d, th3d, q_ddot[0], q_ddot[1], q_ddot[2]]


def main():
    t_eval = np.linspace(0, T, 2000)

    last_t = [0.0]
    with tqdm(total=T, desc="Simulating", unit="s", unit_scale=True) as pbar:
        def dynamics_with_progress(t, state):
            pbar.update(t - last_t[0])
            last_t[0] = t
            return dynamics(t, state)
        sol = solve_ivp(dynamics_with_progress, (0, T), state0, t_eval=t_eval, method='RK45',
                        rtol=1e-8, atol=1e-10)

    th1, th2, th3 = sol.y[0], sol.y[1], sol.y[2]
    # ---- Angle plot ----
    plot_simulation_angles(sol.t, th1, th2, th3, theta1_0, theta2_0, theta3_0, filename_angles, should_save_animation)
    
    # ---- Finger animation ----
    _lf_s     = (force_s1, force_s2, force_s3)          if should_apply_link_forces else None
    _lf_mag   = (F_link1, F_link2, F_link3)              if should_apply_link_forces else None
    _lf_alpha = (alpha_link1, alpha_link2, alpha_link3)  if should_apply_link_forces else None
    _ = animate_finger_simulation(sol, l1, l2, l3, speed=simulation_speed,
                                  link_force_s=_lf_s, link_force_mag=_lf_mag, link_force_alpha=_lf_alpha)

    if should_save_animation:
        print(f"Saving animation to {filename_animation} ...")
        anim_save = animate_finger_simulation(sol, l1, l2, l3, speed=simulation_speed, save_fps=30,
                                              link_force_s=_lf_s, link_force_mag=_lf_mag, link_force_alpha=_lf_alpha)
        anim_save.save(filename_animation, writer='pillow', fps=30, dpi=150)
        plt.close(anim_save._fig)
        print("Animation saved.")

    plt.show()


if __name__ == "__main__":
    main()