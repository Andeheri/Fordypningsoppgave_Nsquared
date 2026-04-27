
import numpy as np
from numpy import pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

"""
Mass-spring-damper parameter identifier for a DC motor load

Identified parameter vector:
    theta_star = [Jm + m, Bm + c, k]

SPM:
    z = theta_star^T Phi

with
    Lambda(s) = (s + lambda1)(s + lambda0)
    z         = (1 / Lambda(s)) * Kt * Ia(s)
    Phi       = [s^2/Lambda(s) * theta,
                 s  /Lambda(s) * theta,
                 1  /Lambda(s) * theta]

Estimator:
    Continuous-time least squares with normalization
"""

# ------------------------------------------------------------------
# Define motor parameters
# ------------------------------------------------------------------
V_supply = 12.0
pwm_frequency = 1000.0
pwm_period = 1.0 / pwm_frequency
voltage_turnoff = 1.0

g = 9.81
Ia_stall = 4.9
tau_stall = 220 * g / 1000

Ia_no_load = 0.2
theta_dot_no_load = 130 * 2 * pi / 60

Kt = tau_stall / Ia_stall
Ra = V_supply / Ia_stall
Bm = Kt * Ia_no_load / theta_dot_no_load
Kb = (V_supply - Ra * Ia_no_load) / theta_dot_no_load

# True plant parameters
Jm = 0.093
La = 0.006
m = 0.5
c = 0.2
k = 1.0

# Only these combined parameters are identifiable from the chosen SPM
M_true = Jm + m
D_true = Bm + c
K_true = k

# ------------------------------------------------------------------
# Initial guesses
# ------------------------------------------------------------------
M_est_0 = 0.25
D_est_0 = 0.05
K_est_0 = 8.0

# Initial covariance for continuous-time LS
P0 = np.diag([500.0, 500.0, 500.0])

# ------------------------------------------------------------------
# PI-PD control parameters
# ------------------------------------------------------------------
Kp1, Ki, Kp2, Kd = 100.0, 3.0, 2.0, 5.0
t0, r_max = 1.0, 2.0

def r(t):
    return r_max * np.sin(2 * np.pi * 0.1 * t) * (t > t0)

# ------------------------------------------------------------------
# Adaptive estimator settings
# ------------------------------------------------------------------
lambda1 = 1.0
lambda0 = 10.0
alpha = 1.0

# ------------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------------
T = 10.0
t_eval = np.linspace(0, T, 5000)

save_folder = "adaptive_control/figures"
filename = "msd_parameter_estimates_ctls.png"

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def voltage_clamp(V):
    return np.clip(V, -V_supply, V_supply)

def control_law(theta_angle, theta_err, omega, e_int):
    V = Kp1 * theta_err + Ki * e_int - (Kp2 * theta_angle + Kd * omega)
    return voltage_clamp(V)

def pwm_voltage(t, V_ideal):
    d = np.abs(V_ideal) / V_supply
    phase = (t % pwm_period) / pwm_period
    active = np.sign(V_ideal) * V_supply
    pwm = np.where(phase < d, active, 0.0)
    return np.where(np.abs(V_ideal) < voltage_turnoff, 0.0, pwm)

# ------------------------------------------------------------------
# Filter helpers for 1 / Lambda(s)
#
# Realization:
#   g_dot = -lambda0 * g + u
#   h_dot = -lambda1 * h + g
#
# Then:
#   h                               = (1 / Lambda(s)) * u
#   g - lambda1 * h                 = (s / Lambda(s)) * u
#   u - (lambda0 + lambda1)*(g - lambda1*h) - lambda0*lambda1*h
#                                   = (s^2 / Lambda(s)) * u
# ------------------------------------------------------------------
def lambda_filter_rhs(u, g_state, h_state):
    dg = -lambda0 * g_state + u
    dh = -lambda1 * h_state + g_state
    return dg, dh

def lambda_filter_outputs(u, g_state, h_state):
    filt_0 = h_state
    filt_1 = g_state - lambda1 * h_state
    filt_2 = u - (lambda0 + lambda1) * filt_1 - lambda0 * lambda1 * filt_0
    return filt_0, filt_1, filt_2

# ------------------------------------------------------------------
# Combined dynamics
#
# State vector:
# x = [theta_angle, omega, i_a, e_int,
#      g_z, h_z, g_theta, h_theta,
#      M_hat, D_hat, K_hat,
#      P11, P12, P13, P21, P22, P23, P31, P32, P33]
# ------------------------------------------------------------------
def dynamics(t, x):
    theta_angle, omega, i_a, e_int = x[0:4]
    g_z, h_z, g_theta, h_theta = x[4:8]
    param_hat = x[8:11]
    P = x[11:20].reshape(3, 3)

    # Control
    theta_err = r(t) - theta_angle
    V_ideal = control_law(theta_angle, theta_err, omega, e_int)
    V = pwm_voltage(t, V_ideal)

    # True plant
    dtheta = omega
    domega = (Kt * i_a - D_true * omega - K_true * theta_angle) / M_true
    di_a = (V - Ra * i_a - Kb * omega) / La
    de_int = theta_err

    # Filter z = (1 / Lambda) * Kt * Ia
    dg_z, dh_z = lambda_filter_rhs(Kt * i_a, g_z, h_z)
    z, _, _ = lambda_filter_outputs(Kt * i_a, g_z, h_z)

    # Filter theta to build Phi
    dg_theta, dh_theta = lambda_filter_rhs(theta_angle, g_theta, h_theta)
    phi_0, phi_1, phi_2 = lambda_filter_outputs(theta_angle, g_theta, h_theta)
    Phi = np.array([phi_2, phi_1, phi_0])

    # Normalized continuous-time LS
    ms2 = 1.0 + alpha * (Phi @ Phi)
    epsilon = (z - param_hat @ Phi) / ms2

    dparam_hat = P @ (Phi * epsilon)
    dP = -(P @ np.outer(Phi, Phi) @ P) / ms2

    # Simple positivity projection
    mins = np.array([1e-5, 1e-5, 1e-5])
    for i in range(3):
        if param_hat[i] <= mins[i] and dparam_hat[i] < 0:
            dparam_hat[i] = 0.0

    return [
        dtheta, domega, di_a, de_int,
        dg_z, dh_z, dg_theta, dh_theta,
        dparam_hat[0], dparam_hat[1], dparam_hat[2],
        *dP.reshape(-1)
    ]


if __name__ == "__main__":
    x0 = [
        0.0, 0.0, 0.0, 0.0,   # theta_angle, omega, i_a, e_int
        0.0, 0.0, 0.0, 0.0,   # g_z, h_z, g_theta, h_theta
        M_est_0, D_est_0, K_est_0,
        *P0.reshape(-1)
    ]

    with tqdm(
        total=T,
        desc="Simulating",
        unit="s",
        bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} s [{elapsed}<{remaining}]"
    ) as pbar:
        last_t = [0.0]

        def dynamics_tracked(t, x):
            pbar.update(t - last_t[0])
            last_t[0] = t
            return dynamics(t, x)

        sol = solve_ivp(
            dynamics_tracked,
            [0, T],
            x0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
            max_step=2e-3,   # important for resolving PWM switching
        )

    theta_sol = sol.y[0]
    omega_sol = sol.y[1]
    i_a_sol = sol.y[2]
    e_int_sol = sol.y[3]

    M_hat_sol = sol.y[8]
    D_hat_sol = sol.y[9]
    K_hat_sol = sol.y[10]

    # Compute filtered signals again for offline LS sanity check
    g_z_sol = sol.y[4]
    h_z_sol = sol.y[5]
    g_th_sol = sol.y[6]
    h_th_sol = sol.y[7]

    z_vec = h_z_sol
    phi_0 = h_th_sol
    phi_1 = g_th_sol - lambda1 * h_th_sol
    phi_2 = theta_sol - (lambda0 + lambda1) * phi_1 - lambda0 * lambda1 * phi_0

    Phi_mat = np.vstack([phi_2, phi_1, phi_0]).T
    theta_ls = np.linalg.lstsq(Phi_mat, z_vec, rcond=None)[0]

    print("Final adaptive estimate:")
    print(f"  M_hat = {M_hat_sol[-1]:.6f}   (true {M_true:.6f})")
    print(f"  D_hat = {D_hat_sol[-1]:.6f}   (true {D_true:.6f})")
    print(f"  K_hat = {K_hat_sol[-1]:.6f}   (true {K_true:.6f})")

    print("\nOffline LS on the same filtered data:")
    print(f"  M_ls  = {theta_ls[0]:.6f}")
    print(f"  D_ls  = {theta_ls[1]:.6f}")
    print(f"  K_ls  = {theta_ls[2]:.6f}")

    # Voltage signals for plotting
    V_ideal = control_law(theta_sol, r(t_eval) - theta_sol, omega_sol, e_int_sol)

    t_pwm = np.linspace(0, T, 20 * int(T * pwm_frequency))
    V_ideal_dense = np.interp(t_pwm, t_eval, V_ideal)
    pwm_signal = pwm_voltage(t_pwm, V_ideal_dense)

    samples_per_period = 20
    kernel = np.ones(samples_per_period) / samples_per_period
    pwm_average = np.convolve(pwm_signal, kernel, mode="same")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def setup_axis(ax, ylabel, color="black", title=None):
        if title:
            ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid()

    fig, axes = plt.subplots(
        5, 1, figsize=(12, 16),
        gridspec_kw={"height_ratios": [1.3, 1.3, 1.3, 2.0, 1.5]}
    )

    axes[0].plot(t_eval, M_hat_sol, label=r"$\widehat{J_m + m}$")
    axes[0].axhline(M_true, color="r", linestyle="--", label=rf"$J_m+m$ = {M_true:.4f}")
    axes[0].axhline(theta_ls[0], color="g", linestyle=":", label=rf"LS = {theta_ls[0]:.4f}")
    axes[0].legend()
    setup_axis(axes[0], "Inertia term", title="Estimate of $(J_m + m)$")

    axes[1].plot(t_eval, D_hat_sol, label=r"$\widehat{B_m + c}$")
    axes[1].axhline(D_true, color="r", linestyle="--", label=rf"$B_m+c$ = {D_true:.4f}")
    axes[1].axhline(theta_ls[1], color="g", linestyle=":", label=rf"LS = {theta_ls[1]:.4f}")
    axes[1].legend()
    setup_axis(axes[1], "Damping term", title="Estimate of $(B_m + c)$")

    axes[2].plot(t_eval, K_hat_sol, label=r"$\hat{k}$")
    axes[2].axhline(K_true, color="r", linestyle="--", label=rf"$k$ = {K_true:.4f}")
    axes[2].axhline(theta_ls[2], color="g", linestyle=":", label=rf"LS = {theta_ls[2]:.4f}")
    axes[2].legend()
    setup_axis(axes[2], "Spring term", title="Estimate of $k$")

    axes[3].plot(t_eval, theta_sol, label=r"$\theta$ (rad)")
    axes[3].plot(t_eval, r(t_eval), "r--", label="Reference (rad)")
    axes[3].legend()
    setup_axis(axes[3], "Angle (rad)", title="Joint Angle vs Reference")

    axes[4].plot(t_pwm, pwm_signal, color="tab:orange", label="V PWM (V)", linewidth=0.5, alpha=0.5)
    axes[4].plot(t_eval, V_ideal, color="tab:green", label="V ideal (V)", linewidth=1.5)
    axes[4].plot(t_pwm, pwm_average, color="tab:red", label="V avg (V)", linewidth=1.5, linestyle="--")
    axes[4].plot(t_eval, i_a_sol, color="tab:blue", label=r"$i_a$ (A)", linewidth=1.2)
    axes[4].axhline(V_supply, color="gray", linestyle=":", label=f"V_supply = ±{V_supply} V")
    axes[4].axhline(-V_supply, color="gray", linestyle=":")
    axes[4].axhline(voltage_turnoff, color="purple", linestyle="--", linewidth=1.0, label=f"V_turnoff = ±{voltage_turnoff} V")
    axes[4].axhline(-voltage_turnoff, color="purple", linestyle="--", linewidth=1.0)
    axes[4].legend(loc="upper left", ncol=2)
    setup_axis(axes[4], "Voltage / Current", title="PWM Control Signal and Armature Current")

    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(f"{save_folder}/{filename}", dpi=150)
    plt.show()
