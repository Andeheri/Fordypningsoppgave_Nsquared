import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from visualization_finger_simulation import animate_finger_simulation, plot_simulation_angles
from homogeneous_transformations import homogeneous_transform

"""
----------------------- Define Constants -----------------------
"""
m1, m2, m3 = 0.02, 0.015, 20  # Masses [kg]
l1, l2, l3 = 0.048, 0.030, 0.024  # Lengths [m]
r1, r2, r3 = 0.085, 0.085, 0.085  # Radii [m]

k1, k2, k3 = 10.0, 10.0, 10.0  # Spring constants [N*m/rad]
b1, b2, b3 = 2.0, 2.0, 2.0  # Damping coefficients [N*m*s/rad]

theta1_0, theta2_0, theta3_0 = pi/6, pi/4, pi/12  # Spring rest angles [rad]

should_apply_external_force = True  # Set to True to apply external force
F_ext_magnitude = 250.0  # Magnitude of external force applied at the fingertip [N]
F_ext_target = np.array([1.0, 0.0])  # Target point the force pulls toward [m]

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


def external_force(theta1, theta2, theta3, u, target=None):
    """
    Computes the 3x1 vector of generalized torques due to an external force of
    magnitude u applied at the fingertip (origin of HT i=3), directed toward
    `target` (default: global origin).

    Uses the virtual-work principle: tau_ext = J^T @ F, where J is the 2x3
    positional Jacobian of the tip and F = u * (target - p) / ||target - p||.

    Parameters
    ----------
    theta1, theta2, theta3 : joint angles [rad]
    u                      : force magnitude [N]
    target                 : 2-element array-like, target point [m] (default (0, 0))

    Returns
    -------
    tau_ext : np.ndarray, shape (3,)
    """
    if target is None:
        target = np.zeros(2)
    target = np.asarray(target, dtype=float)

    # Tip position in global coordinates (translation column of HT for i=3)
    HT = homogeneous_transform(theta1, theta2, theta3, l1, l2, l3, i=3)
    p = HT[:2, 2]  # shape (2,)

    # Force vector directed from tip toward target
    delta = target - p
    delta_norm = np.linalg.norm(delta)
    if delta_norm < 1e-12:
        return np.zeros(3)
    F = u * (delta / delta_norm)  # shape (2,)

    # 2x3 positional Jacobian of the tip
    # p = R(t1)*[l1,0] + R(t1+t2)*[l2,0] + R(t1+t2+t3)*[l3,0]
    # dp/dti has columns: partial derivatives with respect to each cumulative angle
    t1  = theta1
    t12 = theta1 + theta2
    t123 = theta1 + theta2 + theta3
    J = np.array([
        [-l1*sin(t1) - l2*sin(t12) - l3*sin(t123),  -l2*sin(t12) - l3*sin(t123),  -l3*sin(t123)],
        [ l1*cos(t1) + l2*cos(t12) + l3*cos(t123),   l2*cos(t12) + l3*cos(t123),   l3*cos(t123)],
    ])  # shape (2, 3)

    return J.T @ F  # shape (3,)



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
    tau_ext = external_force(th1, th2, th3, F_ext_magnitude, F_ext_target) if should_apply_external_force else np.zeros(3)

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
    _fmag    = F_ext_magnitude if should_apply_external_force else None
    _ftarget = F_ext_target    if should_apply_external_force else None
    _ = animate_finger_simulation(sol, l1, l2, l3, speed=simulation_speed, force_magnitude=_fmag, force_target=_ftarget)

    if should_save_animation:
        print(f"Saving animation to {filename_animation} ...")
        anim_save = animate_finger_simulation(sol, l1, l2, l3, speed=simulation_speed, save_fps=30, force_magnitude=_fmag, force_target=_ftarget)
        anim_save.save(filename_animation, writer='pillow', fps=30, dpi=150)
        plt.close(anim_save._fig)
        print("Animation saved.")

    plt.show()


if __name__ == "__main__":
    main()