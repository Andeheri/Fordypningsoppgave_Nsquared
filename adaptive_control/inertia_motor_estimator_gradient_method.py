import numpy as np
from numpy import pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ─── Motor parameters (Pololu 25D, 12 V) ─────────────────────────────────────
V_supply          = 12.0
pwm_frequency     = 1000.0           # Hz
pwm_period        = 1.0 / pwm_frequency
voltage_turnoff   = 1.0              # V – suppress switching at near-zero voltages

g                 = 9.81
Ia_stall          = 4.9              # A
tau_stall         = 220 * g / 1000   # Nm

Ia_no_load        = 0.2              # A
theta_dot_no_load = 130 * 2 * pi / 60  # rad/s

Kt = tau_stall / Ia_stall            # Nm/A  ≈ 0.44
Ra = V_supply / Ia_stall             # Ω     ≈ 2.44
Bm = Kt * Ia_no_load / theta_dot_no_load
Kb = (V_supply - Ra * Ia_no_load) / theta_dot_no_load

Jm = 0.093   # kg·m²  (true rotor inertia – unknown to estimator)
La = 0.006   # H

# ─── PI-PD controller ────────────────────────────────────────────────────────
Kp1, Ki, Kp2, Kd = 50.0, 3.0, 2.0, 5.0
t0, r_max = 0.1, 5.0
r = lambda t: r_max * (t > t0)   # step reference [rad]

# ─── Adaptive estimator ───────────────────────────────────────────────────────
# Mechanical equation:
#
#   Jm·θ_ddot + Bm·θ_dot = Kt·Ia
#
# Filtered by 1/Λ, with Λ(s) = (s + λ1)(s + λ0):
#
#   z  = 1/Λ [Kt·Ia]  −  s/Λ [Bm·θ]
#   Φ  = s²/Λ [θ]
#   θ* = Jm
#
#   SPM: z = Jm · Φ
#
# Gradient estimator:
#   dJm_est/dt = Γ · ε · Φ,   ε = z − Jm_est · Φ
#
# Filter realisations:
#
# For input x:
#   f0_dot = −λ0·f0 + x
#   f1_dot = −λ1·f1 + f0
#
# Then:
#   f1 = 1/Λ [x]
#
# For θ:
#   1/Λ [θ]    = f_theta_1
#   s/Λ [θ]    = f_theta_0 − λ1·f_theta_1
#   s²/Λ [θ]   = θ − (λ0 + λ1)·s/Λ[θ] − λ0λ1·1/Λ[θ]

lambda0   = 5.0
lambda1   = 10.0
Gamma     = 1000.0
Jm_est_0  = 0.02    # initial estimate [kg·m²]

# ─── Simulation parameters ────────────────────────────────────────────────────
T      = 0.5
t_eval = np.linspace(0, T, 5_000)

save_folder = "adaptive_control/figures"
filename    = "inertia_estimator_gradient_results.png"


# ─── Helper functions ─────────────────────────────────────────────────────────
def voltage_clamp(V):
    return np.clip(V, -V_supply, V_supply)


def control_law(theta_angle, theta_err, omega, e_int):
    V = Kp1 * theta_err + Ki * e_int - (Kp2 * theta_angle + Kd * omega)
    return voltage_clamp(V)


def pwm_voltage(t, V_ideal):
    d      = np.abs(V_ideal) / V_supply
    phase  = (t % pwm_period) / pwm_period
    active = np.sign(V_ideal) * V_supply
    pwm    = np.where(phase < d, active, 0.0)
    return np.where(np.abs(V_ideal) < voltage_turnoff, 0.0, pwm)


# ─── Combined dynamics ────────────────────────────────────────────────────────
# State vector x = [theta_angle, omega, i_a, e_int,
#                   f_ktia_0, f_ktia_1,
#                   f_theta_0, f_theta_1,
#                   Jm_est]
#
#                    0            1      2    3
#                    4          5
#                    6          7
#                    8

def dynamics(t, x):
    theta_angle, omega, i_a, e_int = x[0], x[1], x[2], x[3]
    f_ktia_0, f_ktia_1             = x[4], x[5]
    f_theta_0, f_theta_1           = x[6], x[7]
    Jm_est                         = x[8]

    # Control
    theta_err = r(t) - theta_angle
    V_ideal   = control_law(theta_angle, theta_err, omega, e_int)
    V         = pwm_voltage(t, V_ideal)

    # Motor dynamics
    dtheta = omega
    domega = (Kt * i_a - Bm * omega) / Jm
    di_a   = (V - Ra * i_a - Kb * omega) / La
    de_int = theta_err

    # ─── Second-order filters ────────────────────────────────────────────────
    # 1/Λ [Kt·Ia]
    df_ktia_0 = -lambda0 * f_ktia_0 + Kt * i_a
    df_ktia_1 = -lambda1 * f_ktia_1 + f_ktia_0

    # 1/Λ [θ]
    df_theta_0 = -lambda0 * f_theta_0 + theta_angle
    df_theta_1 = -lambda1 * f_theta_1 + f_theta_0

    # Filtered SPM signals
    one_over_Lambda_KtIa = f_ktia_1
    one_over_Lambda_theta = f_theta_1

    s_over_Lambda_theta = f_theta_0 - lambda1 * f_theta_1

    s2_over_Lambda_theta = (
        theta_angle
        - (lambda0 + lambda1) * s_over_Lambda_theta
        - lambda0 * lambda1 * one_over_Lambda_theta
    )

    z   = one_over_Lambda_KtIa - Bm * s_over_Lambda_theta
    Phi = s2_over_Lambda_theta

    # Estimator
    epsilon = z - Jm_est * Phi
    dJm_est = Gamma * epsilon * Phi

    return [
        dtheta,
        domega,
        di_a,
        de_int,
        df_ktia_0,
        df_ktia_1,
        df_theta_0,
        df_theta_1,
        dJm_est,
    ]


if __name__ == "__main__":
    x0 = [
        0.0, 0.0, 0.0, 0.0,     # theta_angle, omega, i_a, e_int
        0.0, 0.0,               # f_ktia_0, f_ktia_1
        0.0, 0.0,               # f_theta_0, f_theta_1
        Jm_est_0,               # Jm_est
    ]

    with tqdm(
        total=T,
        desc='Simulating',
        unit='s',
        bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} s [{elapsed}<{remaining}]'
    ) as pbar:
        last_t = [0.0]

        def dynamics_tracked(t, x):
            pbar.update(t - last_t[0])
            last_t[0] = t
            return dynamics(t, x)

        sol = solve_ivp(
            dynamics_tracked, [0, T], x0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6, atol=1e-9,
        )

    theta_sol   = sol.y[0]
    omega_sol   = sol.y[1]
    i_a_sol     = sol.y[2]
    e_int_sol   = sol.y[3]
    Jm_est_sol  = sol.y[8]

    # Compute voltage signals for plotting
    V_ideal = control_law(theta_sol, r(t_eval) - theta_sol, omega_sol, e_int_sol)

    # Dense time grid for PWM: 20 samples per PWM period
    t_pwm          = np.linspace(0, T, 20 * int(T * pwm_frequency))
    V_ideal_dense  = np.interp(t_pwm, t_eval, V_ideal)
    pwm_signal     = pwm_voltage(t_pwm, V_ideal_dense)

    samples_per_period = 20
    kernel      = np.ones(samples_per_period) / samples_per_period
    pwm_average = np.convolve(pwm_signal, kernel, mode='same')

    # ─── Plot style parameters ────────────────────────────────────────────────
    TITLE_SIZE  = 14
    LABEL_SIZE  = 13
    TICK_SIZE   = 12
    LEGEND_SIZE = 11
    LINE_WIDTH  = 1.8

    # ─── Plots ────────────────────────────────────────────────────────────────
    def setup_axis(ax, ylabel, color='black', title=None, show_xlabel=False):
        if title:
            ax.set_title(title, fontsize=TITLE_SIZE)
        if show_xlabel:
            ax.set_xlabel('Time (s)', fontsize=LABEL_SIZE)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False)
        ax.set_ylabel(ylabel, color=color, fontsize=LABEL_SIZE)
        ax.tick_params(axis='y', labelcolor=color, labelsize=TICK_SIZE)
        ax.tick_params(axis='x', labelsize=TICK_SIZE)
        ax.grid()

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1.5, 1.5]})

    axes[0].plot(t_eval, Jm_est_sol, label=r'$\hat{J}_m$', linewidth=LINE_WIDTH)
    axes[0].axhline(Jm, color='r', linestyle='--', label=rf'$J_m$ = {Jm:.4f} kg·m$^2$  (true)', linewidth=LINE_WIDTH)
    axes[0].legend(fontsize=LEGEND_SIZE)
    setup_axis(axes[0], r'Inertia [kg·m$^2$]', title='Rotor Inertia Estimate')

    axes[1].plot(t_eval, theta_sol, label=r'$\theta$ (rad)', linewidth=LINE_WIDTH)
    axes[1].legend(fontsize=LEGEND_SIZE)
    setup_axis(axes[1], 'Angle [rad]')

    ax_v = axes[2]
    ax_i = ax_v.twinx()
    ax_v.plot(t_eval, V_ideal, color='tab:orange', label='$V$ [V]',      linewidth=LINE_WIDTH)
    ax_i.plot(t_eval, i_a_sol, color='tab:blue',   label=r'$i_a$ [A]',  linewidth=LINE_WIDTH)

    # Zero-aligned limits with +1 A padding on current
    i_hi = max(i_a_sol) + 1.0
    v_hi = max(V_ideal)  + 1.0
    i_lo_raw = min(i_a_sol) - 0.5
    v_lo_raw = min(V_ideal) - 1.0
    p_i = -i_lo_raw / (i_hi - i_lo_raw)
    p_v = -v_lo_raw / (v_hi - v_lo_raw)
    p   = max(p_i, p_v)
    ax_i.set_ylim(p * i_hi / (p - 1), i_hi)
    ax_v.set_ylim(p * v_hi / (p - 1), v_hi)

    lines_v, labels_v = ax_v.get_legend_handles_labels()
    lines_i, labels_i = ax_i.get_legend_handles_labels()
    ax_v.legend(lines_v + lines_i, labels_v + labels_i, loc='upper left', fontsize=LEGEND_SIZE)
    setup_axis(ax_v, 'Voltage [V]', color='tab:orange', show_xlabel=True)
    ax_i.set_ylabel('Current [A]', color='tab:blue', fontsize=LABEL_SIZE)
    ax_i.tick_params(axis='y', labelcolor='tab:blue', labelsize=TICK_SIZE)

    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(f"{save_folder}/{filename}", dpi=150)