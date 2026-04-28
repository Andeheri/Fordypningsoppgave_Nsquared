import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from visualization_finger_simulation_v2 import animate_finger_simulation, plot_simulation_angles, _move_to_secondary
from dynamics import *
"""
----------------------- Motor dynamics -----------------------
"""

import numpy as np
from numpy import sin, pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov
from tqdm import tqdm
import os

"""
Define motor parameters
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

# PWM parameters
pwm_frequency   = 1000.0          # Hz
pwm_period      = 1.0 / pwm_frequency
voltage_turnoff = 1.0             # V – suppress switching at near-zero voltages

Ia_max = Ia_stall  # A (max current magnitude, for clamping)

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


def current_clamp(i_cmd):
    return float(np.clip(i_cmd, -Ia_stall, Ia_stall))


def control_law(x: np.ndarray, r_val: float, K: np.ndarray, L: float):
    i_cmd = -K @ x + L * r_val
    return current_clamp(i_cmd)


def closed_loop_dynamics(t, z):
    """
    State:
    z = [theta, omega, i_a, e_int_curr, theta_m, omega_m, K1, K2, L]

    Architecture:
      - MRAC outer loop  : computes desired current i_cmd = -K@x + L*r
      - PI inner loop    : drives i_a → i_cmd via voltage command
      - PWM              : converts voltage command to switched supply voltage
      - Electrical plant : di_a = (V_pwm - Ra*i_a - Kb*omega) / La
      - Mechanical plant : driven by actual i_a (not i_cmd)
      - Adaptive law     : sees i_cmd as the nominal input (two-timescale separation)
    """
    theta, omega = z[0], z[1]
    xm = z[2:4]
    K  = z[4:6]
    L  = z[6]

    x   = np.array([theta, omega])
    r_t = r(t)
    e   = x - xm

    # MRAC outer loop: desired current command
    i_cmd = control_law(x, r_t, K, L)

    # Mechanical plant (driven by actual current i_a)
    dtheta = omega
    domega = float(A_true[1, 0] * theta + A_true[1, 1] * omega + B_true[1, 0] * i_cmd + d_true(0)[1])

    # Reference model
    dxm = A_m @ xm + B_m.flatten() * r_t

    # MRAC adaptive law (uses i_cmd as nominal control input)
    sigma = (B_m.T @ P @ e.reshape(-1, 1)).item()
    dK    = Gamma_K @ x * sigma
    dL    = -gamma_L * r_t * sigma

    return np.hstack(([dtheta, domega], dxm, dK, [dL]))


if __name__ == "__main__":
    """
    Simulation parameters
    """
    T = 10.0
    N = 5000
    t_eval = np.linspace(0, T, N)

    # [theta, omega, theta_m, omega_m, K1, K2, L]
    z0 = [
        0.0, 0.0,              # plant: theta, omega
        0.0, 0.0,              # reference model
        K_0[0], K_0[1],        # adaptive K
        L_0                    # adaptive L
    ]

    save_folder = "adaptive_control/figures"
    filename = "mrac_without_mass_spring_damper"
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

    print("Adaptive gains at final time:")
    print(f"K_0 = [{float(K1[-1])}, {float(K2[-1])}]")
    print(f"L_0 = {L[-1]}")

    x_all  = sol.y[0:2]
    r_all  = r(t_eval)
    i_cmd  = np.clip(-(K1 * x_all[0] + K2 * x_all[1]) + L * r_all, -Ia_stall, Ia_stall)

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)

    def setup_axis(ax, ylabel, title=None):
        if title:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    # Theta tracking
    axes[0].plot(t_eval, theta_m, '--', label=r'$\theta_m$')
    axes[0].plot(t_eval, theta, label=r'$\theta$')
    axes[0].plot(t_eval, r_all, 'r--', alpha=0.5, label='r')
    axes[0].legend()
    setup_axis(axes[0], 'Angle (rad)', title='Angle Tracking')

    # Current tracking (inner loop)
    axes[1].plot(t_eval, i_cmd, '--', label=r'$i_{cmd}$ (MRAC)')
    axes[1].axhline( Ia_stall, color='gray', linestyle=':', linewidth=0.8)
    axes[1].axhline(-Ia_stall, color='gray', linestyle=':', linewidth=0.8)
    axes[1].legend()
    setup_axis(axes[1], 'Current (A)', title='Current Tracking (PI inner loop)')

    # Adaptive gains
    axes[2].plot(t_eval, K1, label='K1')
    axes[2].plot(t_eval, K2, label='K2')
    axes[2].plot(t_eval, L,  label='L')
    # axes[2].axhline(K_star[0, 0], color='C0', linestyle='--', alpha=0.7, label='K1*')
    # axes[2].axhline(K_star[0, 1], color='C1', linestyle='--', alpha=0.7, label='K2*')
    # axes[2].axhline(L_star,       color='C2', linestyle='--', alpha=0.7, label='L*')
    axes[2].legend()
    setup_axis(axes[2], 'Gain', title='Adaptive Gains')
    axes[2].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(f"{save_folder}/{filename}.png", dpi=300)

    plt.show()
    quit()

# """
# ----------------------- Define Constants -----------------------
# """

# # Initial conditions: all angles and velocities zero
# state0 = [0.0, 0.0, 0.0,  # theta1, theta2, theta3
#           0.0, 0.0, 0.0]  # theta1_dot, theta2_dot, theta3_dot
# T = 2.0  # Total simulation time [s]
# N = 1000  # Number of simulation timesteps
# simulation_speed = 1.0  # Playback speed multiplier (0.5 = half speed, 1.0 = real-time, etc.)

# # ---- Link force configuration ----------------------------------------
# # Attachment positions: distance from the proximal joint of each link [m]
# # Allowed ranges:  0 <= force_s1 <= l1,  0 <= force_s2 <= l2,  0 <= force_s3 <= l3
# force_s1 = l1 * 0.5   # link 1 – circle mounted at midpoint
# force_s2 = l2 * 0.5   # link 2 – circle mounted at midpoint
# force_s3 = l3 * 0.5   # link 3 – circle mounted at midpoint

# # Circle radii mounted at each link midpoint [m]
# # The pulling force is applied at the circumference of the circle,
# # offset perpendicularly (radially outward, 90° CCW from link axis) from the midpoint.
# r_circle1 = 0.010   # link 1 – circle radius [m]
# r_circle2 = 0.010   # link 2 – circle radius [m]
# r_circle3 = 0.010   # link 3 – circle radius [m]

# # ---- Whiffle tree configuration -----------------------------------------
# # One motor pulls a whiffle tree that distributes force to links 1 and 2.
# # wt_frac1 + wt_frac2 must equal 1.  (e.g. 2/3 to link 1, 1/3 to link 2)
# wt_frac1 = 2/3   # fraction of motor force delivered to link 1
# wt_frac2 = 1/3   # fraction of motor force delivered to link 2

# # ---- Motor force input u(t) [N] -----------------------------------------
# # This is the total force the motor applies to the whiffle tree.
# # It is distributed to the links as:
# #   F_link1 = wt_frac1 * u(t)
# #   F_link2 = wt_frac2 * u(t)
# def u(t):
#     return 1.0   # Force applied to whiffle tree from motor

# # Force magnitude at each timestep [N] — callables  F(t, state) -> float.
# # Derived from u(t) via the whiffle tree fractions. F_link3 is independent.
# F_link1 = lambda t, state: wt_frac1 * u(t)
# F_link2 = lambda t, state: wt_frac2 * u(t)
# F_link3 = lambda t, state: 0.0   # not driven by the whiffle tree

# # Aim target fractions (0–1): the force on each link points FROM the attachment
# # point TOWARDS a fractional position along the link below it.
# #   0.0 → proximal joint of the link below
# #   1.0 → distal joint of the link below
# # Link 1 aims at the metacarpal (length l0), link 2 at link 1, link 3 at link 2.
# aim_frac1 = 0.5   # link 1 → aims at 0.5 * l0 along the metacarpal
# aim_frac2 = 0.5   # link 2 → aims at 0.5 * l1 along link 1
# aim_frac3 = 0.5   # link 3 → aims at 0.5 * l2 along link 2

# should_apply_link_forces = True

# should_save_animation = True  # Set to True to save the animation as a GIF file
# should_show_plots = False  # Set to False to skip showing plots (useful when only saving the animation)
# note = "prior_to_motor_integration"  # A note to include in the filename for clarity
# save_folder = "adaptive_control/figures/with_finger_dynamics"

# filename_angles    = f"{save_folder}/finger_simulation_angles_{note}.png"  # Filename for the saved angles plot
# filename_animation = f"{save_folder}/finger_simulation_{note}.gif"         # Filename for the saved animation

# """
# ----------------------- Simulation and Visualization -----------------------
# """


# def main():
#     t_eval = np.linspace(0, T, N)

#     # Pack force config for the integrator
#     _force_s   = (force_s1, force_s2, force_s3)       if should_apply_link_forces else None
#     _force_r   = (r_circle1, r_circle2, r_circle3)    if should_apply_link_forces else None
#     _force_F   = (F_link1, F_link2, F_link3)          if should_apply_link_forces else None
#     _force_aim = (aim_frac1, aim_frac2, aim_frac3)    if should_apply_link_forces else None

#     last_t = [0.0]
#     with tqdm(total=T, desc="Simulating", unit="s", unit_scale=True) as pbar:
#         def dynamics_with_progress(t, state):
#             pbar.update(t - last_t[0])
#             last_t[0] = t
#             return dynamics(t, state,
#                             force_s=_force_s, r_circle=_force_r,
#                             F_link=_force_F, aim_frac=_force_aim)
#         sol = solve_ivp(dynamics_with_progress, (0, T), state0, t_eval=t_eval, method='RK45',
#                         rtol=1e-8, atol=1e-10)

#     th1, th2, th3 = sol.y[0], sol.y[1], sol.y[2]

#     # ---- Cable tension from passive finger dynamics ----
#     # T_i > 0: cable i would be pulled taut (motor must provide this tension)
#     # T_i < 0: springs push link away from target (cable would be slack)
#     _ct_args = ((force_s1, force_s2, force_s3),
#                 (r_circle1, r_circle2, r_circle3),
#                 (aim_frac1, aim_frac2, aim_frac3))
#     T_all = np.array([cable_tensions(sol.y[:, i], *_ct_args)
#                       for i in range(sol.y.shape[1])])  # shape (N, 3)

#     # ---- Whiffle tree: back-calculate motor force from T1 and T2 ----------
#     # With T1 = wt_frac1 * F_motor  and  T2 = wt_frac2 * F_motor, the
#     # least-squares estimate of F_motor is:
#     #   F_motor = (wt_frac1*T1 + wt_frac2*T2) / (wt_frac1**2 + wt_frac2**2)
#     _wt_denom = wt_frac1**2 + wt_frac2**2
#     F_motor_wt = (wt_frac1 * T_all[:, 0] + wt_frac2 * T_all[:, 1]) / _wt_denom

#     # ---- Cable / whiffle tree plots (saved silently when should_show_plots=False) ----
#     _orig_backend = plt.get_backend()
#     if not should_show_plots:
#         plt.switch_backend('Agg')

#     fig_ct, (ax_ct, ax_wt) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
#     _move_to_secondary(fig_ct)
#     ax_ct.plot(sol.t, T_all[:, 0], label=r'$T_1$ (link 1)')
#     ax_ct.plot(sol.t, T_all[:, 1], label=r'$T_2$ (link 2)')
#     ax_ct.plot(sol.t, T_all[:, 2], label=r'$T_3$ (link 3)')
#     ax_ct.axhline(0, color='k', linewidth=0.7, linestyle=':')
#     ax_ct.set_ylabel('Cable tension [N]')
#     ax_ct.set_title('Cable tensions from passive finger dynamics')
#     ax_ct.legend()
#     ax_ct.grid(True, alpha=0.3)

#     ax_wt.plot(sol.t, F_motor_wt, color='tab:purple',
#                label=rf'$F_{{motor}}$ (whiffle tree {wt_frac1:.2g}/{wt_frac2:.2g})')
#     ax_wt.axhline(0, color='k', linewidth=0.7, linestyle=':')
#     ax_wt.set_xlabel('Time [s]')
#     ax_wt.set_ylabel('Motor force [N]')
#     ax_wt.set_title('Whiffle tree motor force (back-calculated from $T_1$, $T_2$)')
#     ax_wt.legend()
#     ax_wt.grid(True, alpha=0.3)

#     plt.tight_layout()
#     if should_save_animation:
#         ct_filepath = f"{save_folder}/cable_tensions_{note}.png"
#         os.makedirs(os.path.dirname(os.path.abspath(ct_filepath)), exist_ok=True)
#         fig_ct.savefig(ct_filepath, dpi=300)
#     plt.close(fig_ct)

#     # ---- Angle plot ----
#     plot_simulation_angles(sol.t, th1, th2, th3, theta1_0, theta2_0, theta3_0, filename_angles, should_save_animation)

#     # Restore original backend before creating the animation
#     if not should_show_plots:
#         plt.switch_backend(_orig_backend)
    
#     # ---- Finger animation ----
#     _lf_s   = _force_s
#     _lf_mag = _force_F
#     _lf_r   = _force_r
#     if should_show_plots:
#         _ = animate_finger_simulation(sol, l1, l2, l3, speed=simulation_speed,
#                                       link_force_s=_lf_s, link_force_mag=_lf_mag,
#                                       link_force_r=_lf_r, aim_frac=_force_aim, l0=l0)

#     if should_save_animation:
#         print(f"Saving animation to {filename_animation} ...")
#         anim_save = animate_finger_simulation(sol, l1, l2, l3, speed=simulation_speed, save_fps=30,
#                                               link_force_s=_lf_s, link_force_mag=_lf_mag,
#                                               link_force_r=_lf_r, aim_frac=_force_aim, l0=l0)
#         anim_save.save(filename_animation, writer='pillow', fps=30, dpi=150)
#         plt.close(anim_save._fig)
#         print("Animation saved.")
#     if should_show_plots:
#         plt.show()


# if __name__ == "__main__":
#     main()