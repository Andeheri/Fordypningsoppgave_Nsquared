import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
from tqdm import tqdm
from visualization_finger_simulation_v2 import animate_finger_simulation, plot_simulation_angles, _move_to_secondary
from dynamics import *

"""
----------------------- Motor dynamics -----------------------
"""

V_supply = 12.0   # V
g = 9.81

# Information from datasheet: https://www.pololu.com/file/0J1829/pololu-25d-metal-gearmotors.pdf, page 3
Ia_stall = 4.9  # A
tau_stall = 220 * g / 1000  # Nm (stall torque at 12 V)

Ia_no_load = 0.2  # A
theta_dot_no_load = 130 * 2 * pi / 60  # rad/s

Kt = tau_stall / Ia_stall
Ra = V_supply / Ia_stall
Bm = Kt * Ia_no_load / theta_dot_no_load
Kb = (V_supply - Ra * Ia_no_load) / theta_dot_no_load

# Motor/electrical parameters
Jm = 0.093   # kg*m^2
La = 0.006   # H

r_spindle = 0.01  # m (effective radius of the spindle that the cable winds around)

"""
Reduced 2-state plant:
x = [theta, omega]^T
u = i_cmd   (assume fast inner current loop, so actual current tracks command)
"""

A_true = np.array([
    [0.0, 1.0],
    [0.0, -Bm / Jm]
])

B_true = np.array([
    [0.0],
    [Kt / Jm]
])

d_true = lambda tau: np.array([
    0.0,
    tau / Jm
])

"""
Reference model
Choose desired theta-tracking dynamics here
"""
omega_n = 8.0
zeta = 1.0

A_m = np.array([
    [0.0, 1.0],
    [-omega_n**2, -2.0 * zeta * omega_n]
])

B_m = np.array([
    [0.0],
    [omega_n**2]
])

Q = np.diag([10.0, 1.0])
P = solve_continuous_lyapunov(A_m.T, -Q)

print("P matrix:")
print(P)

"""
Ideal matching parameters (only for verification/debugging)
Controller does NOT use these
"""
K_star = np.linalg.pinv(B_true) @ (A_true - A_m)
L_star = float(np.squeeze(np.linalg.pinv(B_true) @ B_m))

K_0 = (np.linalg.pinv(B_true) @ A_m).flatten()
L_0 = float(np.squeeze(np.linalg.pinv(B_m) @ B_m))

K_0 = [0.0, 0.0]
L_0 = 1.0


print("K_star =", K_star)
print("L_star =", L_star)

"""
Adaptive gains
"""
Gamma_K = np.diag([20.0, 3.0])   # adaptation rate for K = [K1, K2]
gamma_L = 20.0 * 1.0                  # adaptation rate for L

"""
Reference input
"""
t0 = 1.0
r_max = pi / 4
r = lambda t: r_max * (t > t0)
r = lambda t: r_max * sin(2 * pi * 0.5 * t)
# Square wave reference:
r = lambda t: r_max * (sin(2 * pi * 0.3 * t) > 0).astype(float)

"""
------------------------ Finger dynamics constants -----------------------
"""

phi1_0 = 0.0
phi2_0 = 0.0
phi3_0 = 0.0

simulation_speed = 1.0  # Playback speed multiplier (0.5 = half speed, 1.0 = real-time, etc.)

# ---- Link force configuration ----------------------------------------
# Attachment positions: distance from the proximal joint of each link [m]
# Allowed ranges:  0 <= force_s1 <= l1,  0 <= force_s2 <= l2,  0 <= force_s3 <= l3
force_s1 = l1 * 0.5   # link 1 – circle mounted at midpoint
force_s2 = l2 * 0.5   # link 2 – circle mounted at midpoint
force_s3 = l3 * 0.5   # link 3 – circle mounted at midpoint

# Circle radii mounted at each link midpoint [m]
# The pulling force is applied at the circumference of the circle,
# offset perpendicularly (radially outward, 90° CCW from link axis) from the midpoint.
r_circle1 = 0.010   # link 1 – circle radius [m]
r_circle2 = 0.010   # link 2 – circle radius [m]
r_circle3 = 0.010   # link 3 – circle radius [m]

# ---- Whiffle tree configuration -----------------------------------------
# One motor pulls a whiffle tree that distributes force to links 1 and 2.
# wt_frac1 + wt_frac2 must equal 1.  (e.g. 2/3 to link 1, 1/3 to link 2)
wt_frac1 = 2/3   # fraction of motor force delivered to link 1
wt_frac2 = 1/3   # fraction of motor force delivered to link 2

# ---- Motor force input u(t) [N] -----------------------------------------
# This is the total force the motor applies to the whiffle tree.
# It is distributed to the links as:
#   F_link1 = wt_frac1 * u(t)
#   F_link2 = wt_frac2 * u(t)
def u(t):
    return 1.0   # Force applied to whiffle tree from motor

# Force magnitude at each timestep [N] — callables  F(t, state) -> float.
# Derived from u(t) via the whiffle tree fractions. F_link3 is independent.
F_link1 = lambda t, state: wt_frac1 * u(t)
F_link2 = lambda t, state: wt_frac2 * u(t)
F_link3 = lambda t, state: 0.0   # not driven by the whiffle tree

# Aim target fractions (0–1): the force on each link points FROM the attachment
# point TOWARDS a fractional position along the link below it.
#   0.0 → proximal joint of the link below
#   1.0 → distal joint of the link below
# Link 1 aims at the metacarpal (length l0), link 2 at link 1, link 3 at link 2.
aim_frac1 = 0.5   # link 1 → aims at 0.5 * l0 along the metacarpal
aim_frac2 = 0.5   # link 2 → aims at 0.5 * l1 along link 1
aim_frac3 = 0.5   # link 3 → aims at 0.5 * l2 along link 2

_force_s   = (force_s1, force_s2, force_s3)      
_force_r   = (r_circle1, r_circle2, r_circle3)   
_force_aim = (aim_frac1, aim_frac2, aim_frac3)

should_save_animation = True  # Set to True to save the animation as a GIF file
should_show_plots = False  # Set to False to skip showing plots (useful when only saving the animation)
note = "prior_to_motor_integration"  # A note to include in the filename for clarity
save_folder = "adaptive_control/figures/with_finger_dynamics"

filename_angles    = f"{save_folder}/finger_simulation_angles_{note}.png"  # Filename for the saved angles plot
filename_animation = f"{save_folder}/finger_simulation_{note}.gif"         # Filename for the saved animation


def current_clamp(i_cmd):
    return float(np.clip(i_cmd, -Ia_stall, Ia_stall))


def control_law(x: np.ndarray, r_val: float, K: np.ndarray, L: float):
    i_cmd = -K @ x + L * r_val
    return current_clamp(i_cmd)


def closed_loop_dynamics(t, z):
    """
    State:
    z = [theta, omega, theta_m, omega_m, K1, K2, L, phi, phi_dot]

    Architecture:
      - MRAC outer loop  : computes desired current i_cmd = -K@x + L*r
      - Mechanical plant : driven by actual i_cmd
      - Adaptive law     : sees i_cmd as the nominal input (two-timescale separation)
    """
    theta, omega = z[0], z[1]
    xm = z[2:4]
    K  = z[4:6]
    L  = z[6]
    phi_1, phi_2, phi_3 = z[7:10]
    phi_1_dot, phi_2_dot, phi_3_dot = z[10:13]

    x   = np.array([theta, omega])
    r_t = r(t)
    e   = x - xm

    # MRAC outer loop: desired current command
    i_cmd = control_law(x, r_t, K, L)
    output_force = Kt * i_cmd / r_spindle

    # Finger dynamics
    M_mat = M(phi_1, phi_2, phi_3)
    C_mat = C(phi_1, phi_2, phi_3, phi_1_dot, phi_2_dot, phi_3_dot)
    tau_k = Tau_K(phi_1, phi_2, phi_3)
    tau_b = Tau_B(phi_1_dot, phi_2_dot, phi_3_dot)
    _force_F = np.array((wt_frac1 * output_force, wt_frac2 * output_force, 0.0))
    tau_ext = tau_link_forces(phi_1, phi_2, phi_3, t, z[7:13], _force_s, _force_r, _force_F, _force_aim)
    q_dot = np.array([phi_1_dot, phi_2_dot, phi_3_dot])

    # Cable tension the finger's internal model puts on the whiffle tree,
    # i.e. the force the motor would experience from the finger via the cable.
    # NOTE: used as a diagnostic / output variable only — NOT fed into the motor
    # ODE, because cable_tensions() inverts a geometry matrix that becomes singular
    # at reachable finger configurations (e.g. phi2 ≈ π/2), which would crash the
    # solver.  Wire it into domega only once a robust cable-constraint model exists.
    _finger_state = np.array([phi_1, phi_2, phi_3, phi_1_dot, phi_2_dot, phi_3_dot])
    T_cables = cable_tensions(_finger_state, _force_s, _force_r, _force_aim)
    F_motor_from_finger = (wt_frac1 * T_cables[0] + wt_frac2 * T_cables[1]) / (wt_frac1**2 + wt_frac2**2)

    rhs = -C_mat @ q_dot - tau_k - tau_b + tau_ext
    q_ddot = np.linalg.solve(M_mat, rhs)

    # Mechanical plant (driven by actual current i_a)
    dtheta = omega
    domega = float(A_true[1, 0] * theta + A_true[1, 1] * omega + B_true[1, 0] * i_cmd + d_true(0)[1])

    # Reference model
    dxm = A_m @ xm + B_m.flatten() * r_t

    # MRAC adaptive law (uses i_cmd as nominal control input)
    sigma = (B_m.T @ P @ e.reshape(-1, 1)).item()
    dK    = Gamma_K @ x * sigma
    dL    = -gamma_L * r_t * sigma

    return np.hstack(([dtheta, domega], dxm, dK, [dL], q_dot, q_ddot))


def main():
    """
    Simulation parameters
    """
    T = 10.0
    N = 5000
    t_eval = np.linspace(0, T, N)

    # [theta, omega, theta_m, omega_m, K1, K2, L, phi, phi_dot]
    z0 = [
        0.0, 0.0,                # plant: theta, omega
        0.0, 0.0,                # reference model: theta_m, omega_m
        K_0[0], K_0[1],          # adaptive K
        L_0,                     # adaptive L
        phi1_0, phi2_0, phi3_0,  # finger initial angles
        0.0, 0.0, 0.0            # finger initial angular velocities
    ]

    save_folder = "adaptive_control/figures"
    filename = "mrac_with_finger_dynamics"
    os.makedirs(save_folder, exist_ok=True)

    with tqdm(total=T, desc="Simulating", unit="s", dynamic_ncols=True) as pbar:
        last_t = [0.0]

        def ode_with_progress(t, z):
            pbar.update(t - last_t[0])
            last_t[0] = t
            return closed_loop_dynamics(t, z)

        sol = solve_ivp(
            ode_with_progress,
            t_span=[0.0, T],
            y0=z0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )

    # Extract results
    theta   = sol.y[0]
    omega   = sol.y[1]
    theta_m = sol.y[2]
    omega_m = sol.y[3]
    K1      = sol.y[4]
    K2      = sol.y[5]
    L       = sol.y[6]
    phi     = sol.y[7:10]
    phi_dot = sol.y[10:13]

    print("Adaptive gains at final time:")
    print(f"K_0 = [{float(K1[-1])}, {float(K2[-1])}]")
    print(f"L_0 = {L[-1]}")

    x_all  = sol.y[0:2]
    r_all  = r(sol.t)
    i_cmd  = np.clip(-(K1 * x_all[0] + K2 * x_all[1]) + L * r_all, -Ia_stall, Ia_stall)

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)

    def setup_axis(ax, ylabel, title=None):
        if title:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    # Theta tracking
    axes[0].plot(sol.t, theta_m, '--', label=r'$\theta_m$')
    axes[0].plot(sol.t, theta, label=r'$\theta$')
    axes[0].plot(sol.t, r_all, 'r--', alpha=0.5, label='r')
    axes[0].legend()
    setup_axis(axes[0], 'Angle (rad)', title='Angle Tracking')

    # Relative angles of finger joints
    axes[1].axhline(phi1_eq, linestyle='--', label=r'$\phi_{1,\mathrm{eq}}$', color='tab:blue')
    axes[1].plot(sol.t, phi[0], label=r'$\phi_1$', color='tab:blue')
    axes[1].axhline(phi2_eq, linestyle='--', label=r'$\phi_{2,\mathrm{eq}}$', color='tab:orange')
    axes[1].plot(sol.t, phi[1], label=r'$\phi_2$', color='tab:orange')
    axes[1].axhline(phi3_eq, linestyle='--', label=r'$\phi_{3,\mathrm{eq}}$', color='tab:green')
    axes[1].plot(sol.t, phi[2], label=r'$\phi_3$', color='tab:green')
    axes[1].legend()
    setup_axis(axes[1], 'Relative Angle (rad)', title='Finger Joint Angles')

    # Adaptive gains
    axes[2].plot(sol.t, K1, label='K1')
    axes[2].plot(sol.t, K2, label='K2')
    axes[2].plot(sol.t, L,  label='L')
    axes[2].legend()
    setup_axis(axes[2], 'Gain', title='Adaptive Gains')
    axes[2].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(f"{save_folder}/{filename}.png", dpi=300)

    plt.show()

if __name__ == "__main__":
    main()