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
----------------------- Define Constants -----------------------
"""

# Initial conditions: all angles and velocities zero
state0 = [0.0, 0.0, 0.0,  # theta1, theta2, theta3
          0.0, 0.0, 0.0]  # theta1_dot, theta2_dot, theta3_dot
T = 2.0  # Total simulation time [s]
N = 1000  # Number of simulation timesteps
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

should_apply_link_forces = True

should_save_animation = True  # Set to True to save the animation as a GIF file
should_show_plots = False  # Set to False to skip showing plots (useful when only saving the animation)
note = "prior_to_motor_integration"  # A note to include in the filename for clarity
save_folder = "adaptive_control/figures/with_finger_dynamics"

filename_angles    = f"{save_folder}/finger_simulation_angles_{note}.png"  # Filename for the saved angles plot
filename_animation = f"{save_folder}/finger_simulation_{note}.gif"         # Filename for the saved animation

"""
----------------------- Simulation and Visualization -----------------------
"""


def main():
    t_eval = np.linspace(0, T, N)

    # Pack force config for the integrator
    _force_s   = (force_s1, force_s2, force_s3)       if should_apply_link_forces else None
    _force_r   = (r_circle1, r_circle2, r_circle3)    if should_apply_link_forces else None
    _force_F   = (F_link1, F_link2, F_link3)          if should_apply_link_forces else None
    _force_aim = (aim_frac1, aim_frac2, aim_frac3)    if should_apply_link_forces else None

    last_t = [0.0]
    with tqdm(total=T, desc="Simulating", unit="s", unit_scale=True) as pbar:
        def dynamics_with_progress(t, state):
            pbar.update(t - last_t[0])
            last_t[0] = t
            return dynamics(t, state,
                            force_s=_force_s, r_circle=_force_r,
                            F_link=_force_F, aim_frac=_force_aim)
        sol = solve_ivp(dynamics_with_progress, (0, T), state0, t_eval=t_eval, method='RK45',
                        rtol=1e-8, atol=1e-10)

    th1, th2, th3 = sol.y[0], sol.y[1], sol.y[2]

    # ---- Cable tension from passive finger dynamics ----
    # T_i > 0: cable i would be pulled taut (motor must provide this tension)
    # T_i < 0: springs push link away from target (cable would be slack)
    _ct_args = ((force_s1, force_s2, force_s3),
                (r_circle1, r_circle2, r_circle3),
                (aim_frac1, aim_frac2, aim_frac3))
    T_all = np.array([cable_tensions(sol.y[:, i], *_ct_args)
                      for i in range(sol.y.shape[1])])  # shape (N, 3)

    # ---- Whiffle tree: back-calculate motor force from T1 and T2 ----------
    # With T1 = wt_frac1 * F_motor  and  T2 = wt_frac2 * F_motor, the
    # least-squares estimate of F_motor is:
    #   F_motor = (wt_frac1*T1 + wt_frac2*T2) / (wt_frac1**2 + wt_frac2**2)
    _wt_denom = wt_frac1**2 + wt_frac2**2
    F_motor_wt = (wt_frac1 * T_all[:, 0] + wt_frac2 * T_all[:, 1]) / _wt_denom

    # ---- Cable / whiffle tree plots (saved silently when should_show_plots=False) ----
    _orig_backend = plt.get_backend()
    if not should_show_plots:
        plt.switch_backend('Agg')

    fig_ct, (ax_ct, ax_wt) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    _move_to_secondary(fig_ct)
    ax_ct.plot(sol.t, T_all[:, 0], label=r'$T_1$ (link 1)')
    ax_ct.plot(sol.t, T_all[:, 1], label=r'$T_2$ (link 2)')
    ax_ct.plot(sol.t, T_all[:, 2], label=r'$T_3$ (link 3)')
    ax_ct.axhline(0, color='k', linewidth=0.7, linestyle=':')
    ax_ct.set_ylabel('Cable tension [N]')
    ax_ct.set_title('Cable tensions from passive finger dynamics')
    ax_ct.legend()
    ax_ct.grid(True, alpha=0.3)

    ax_wt.plot(sol.t, F_motor_wt, color='tab:purple',
               label=rf'$F_{{motor}}$ (whiffle tree {wt_frac1:.2g}/{wt_frac2:.2g})')
    ax_wt.axhline(0, color='k', linewidth=0.7, linestyle=':')
    ax_wt.set_xlabel('Time [s]')
    ax_wt.set_ylabel('Motor force [N]')
    ax_wt.set_title('Whiffle tree motor force (back-calculated from $T_1$, $T_2$)')
    ax_wt.legend()
    ax_wt.grid(True, alpha=0.3)

    plt.tight_layout()
    if should_save_animation:
        ct_filepath = f"{save_folder}/cable_tensions_{note}.png"
        os.makedirs(os.path.dirname(os.path.abspath(ct_filepath)), exist_ok=True)
        fig_ct.savefig(ct_filepath, dpi=300)
    plt.close(fig_ct)

    # ---- Angle plot ----
    plot_simulation_angles(sol.t, th1, th2, th3, theta1_0, theta2_0, theta3_0, filename_angles, should_save_animation)

    # Restore original backend before creating the animation
    if not should_show_plots:
        plt.switch_backend(_orig_backend)
    
    # ---- Finger animation ----
    _lf_s   = _force_s
    _lf_mag = _force_F
    _lf_r   = _force_r
    if should_show_plots:
        _ = animate_finger_simulation(sol, l1, l2, l3, speed=simulation_speed,
                                      link_force_s=_lf_s, link_force_mag=_lf_mag,
                                      link_force_r=_lf_r, aim_frac=_force_aim, l0=l0)

    if should_save_animation:
        print(f"Saving animation to {filename_animation} ...")
        anim_save = animate_finger_simulation(sol, l1, l2, l3, speed=simulation_speed, save_fps=30,
                                              link_force_s=_lf_s, link_force_mag=_lf_mag,
                                              link_force_r=_lf_r, aim_frac=_force_aim, l0=l0)
        anim_save.save(filename_animation, writer='pillow', fps=30, dpi=150)
        plt.close(anim_save._fig)
        print("Animation saved.")
    if should_show_plots:
        plt.show()


if __name__ == "__main__":
    main()