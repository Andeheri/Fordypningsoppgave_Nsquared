
import numpy as np
from numpy import pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

"""
Fast mass-spring-damper parameter identifier for a DC motor load

This version keeps the plant/controller/reference identical to the
reference script and only changes the simulation backend.

Fast backend:
    - Numba-jitted fixed-step RK4
    - Integrates to each t_eval time exactly
Fallback:
    - solve_ivp with LSODA
"""

# ------------------------------------------------------------------
# User options
# ------------------------------------------------------------------
USE_FAST_SIM = True
SHOW_PLOT = False
SAVE_PLOT = True

# Internal RK4 max step. Smaller = closer to the RK45 reference.
FAST_DT = 1e-4

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
P0 = np.diag([1.0, 1.0, 1.0]) * 100.0

# ------------------------------------------------------------------
# PI-PD control parameters
# ------------------------------------------------------------------
Kp1, Ki, Kp2, Kd = 100.0, 3.0, 2.0, 5.0
t0, r_max = 1.0, 2.0

def r(t):
    return (r_max * (np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.05 * t))) * (t > t0)

# ------------------------------------------------------------------
# Adaptive estimator settings
# ------------------------------------------------------------------
lambda1 = 0.1
lambda0 = 10.0
alpha = 0.0

# ------------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------------
T = 20.0
t_eval = np.linspace(0, T, 5000)

save_folder = "adaptive_control/figures"
filename = "msd_parameter_estimates_fast.png"

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
# Original Python dynamics (fallback solver)
# ------------------------------------------------------------------
def dynamics(t, x):
    theta_angle, omega, i_a, e_int = x[0:4]
    g_z, h_z, g_theta, h_theta = x[4:8]
    param_hat = x[8:11]
    P = x[11:20].reshape(3, 3)

    theta_err = r(t) - theta_angle
    V_ideal = control_law(theta_angle, theta_err, omega, e_int)
    V = pwm_voltage(t, V_ideal)

    dtheta = omega
    domega = (Kt * i_a - D_true * omega - K_true * theta_angle) / M_true
    di_a = (V - Ra * i_a - Kb * omega) / La
    de_int = theta_err

    dg_z, dh_z = lambda_filter_rhs(Kt * i_a, g_z, h_z)
    z, _, _ = lambda_filter_outputs(Kt * i_a, g_z, h_z)

    dg_theta, dh_theta = lambda_filter_rhs(theta_angle, g_theta, h_theta)
    phi_0, phi_1, phi_2 = lambda_filter_outputs(theta_angle, g_theta, h_theta)
    Phi = np.array([phi_2, phi_1, phi_0])

    ms2 = 1.0 + alpha * (Phi @ Phi)
    epsilon = (z - param_hat @ Phi) / ms2

    dparam_hat = P @ (Phi * epsilon)
    dP = -(P @ np.outer(Phi, Phi) @ P) / ms2

    mins = np.array([1e-5, 1e-5, 1e-5])
    for i in range(3):
        if param_hat[i] <= mins[i] and dparam_hat[i] < 0:
            dparam_hat[i] = 0.0

    return np.array([
        dtheta, domega, di_a, de_int,
        dg_z, dh_z, dg_theta, dh_theta,
        dparam_hat[0], dparam_hat[1], dparam_hat[2],
        *dP.reshape(-1)
    ])


# ------------------------------------------------------------------
# Numba dynamics + exact t_eval stepping
# ------------------------------------------------------------------
@njit(cache=True)
def r_nb(t):
    if t > t0:
        return r_max * np.sin(2.0 * np.pi * 0.1 * t)
    return 0.0

@njit(cache=True)
def voltage_clamp_nb(V):
    if V > V_supply:
        return V_supply
    if V < -V_supply:
        return -V_supply
    return V

@njit(cache=True)
def control_law_nb(theta_angle, theta_err, omega, e_int):
    V = Kp1 * theta_err + Ki * e_int - (Kp2 * theta_angle + Kd * omega)
    return voltage_clamp_nb(V)

@njit(cache=True)
def pwm_voltage_nb(t, V_ideal):
    if abs(V_ideal) < voltage_turnoff:
        return 0.0
    d = abs(V_ideal) / V_supply
    phase = (t % pwm_period) / pwm_period
    active = V_supply if V_ideal >= 0.0 else -V_supply
    if phase < d:
        return active
    return 0.0

@njit(cache=True)
def rhs_nb(t, x):
    out = np.empty(20, dtype=np.float64)

    theta_angle = x[0]
    omega = x[1]
    i_a = x[2]
    e_int = x[3]
    g_z = x[4]
    h_z = x[5]
    g_theta = x[6]
    h_theta = x[7]

    param0 = x[8]
    param1 = x[9]
    param2 = x[10]

    P00, P01, P02 = x[11], x[12], x[13]
    P10, P11, P12 = x[14], x[15], x[16]
    P20, P21, P22 = x[17], x[18], x[19]

    theta_err = r_nb(t) - theta_angle
    V_ideal = control_law_nb(theta_angle, theta_err, omega, e_int)
    V = pwm_voltage_nb(t, V_ideal)

    dtheta = omega
    domega = (Kt * i_a - D_true * omega - K_true * theta_angle) / M_true
    di_a = (V - Ra * i_a - Kb * omega) / La
    de_int = theta_err

    dg_z = -lambda0 * g_z + Kt * i_a
    dh_z = -lambda1 * h_z + g_z
    z = h_z

    dg_theta = -lambda0 * g_theta + theta_angle
    dh_theta = -lambda1 * h_theta + g_theta

    phi0 = h_theta
    phi1 = g_theta - lambda1 * h_theta
    phi2 = theta_angle - (lambda0 + lambda1) * phi1 - lambda0 * lambda1 * phi0

    ms2 = 1.0 + alpha * (phi2 * phi2 + phi1 * phi1 + phi0 * phi0)
    epsilon = (z - (param0 * phi2 + param1 * phi1 + param2 * phi0)) / ms2

    tmp0 = phi2 * epsilon
    tmp1 = phi1 * epsilon
    tmp2 = phi0 * epsilon

    dparam0 = P00 * tmp0 + P01 * tmp1 + P02 * tmp2
    dparam1 = P10 * tmp0 + P11 * tmp1 + P12 * tmp2
    dparam2 = P20 * tmp0 + P21 * tmp1 + P22 * tmp2

    if param0 <= 1e-5 and dparam0 < 0.0:
        dparam0 = 0.0
    if param1 <= 1e-5 and dparam1 < 0.0:
        dparam1 = 0.0
    if param2 <= 1e-5 and dparam2 < 0.0:
        dparam2 = 0.0

    v0 = P00 * phi2 + P01 * phi1 + P02 * phi0
    v1 = P10 * phi2 + P11 * phi1 + P12 * phi0
    v2 = P20 * phi2 + P21 * phi1 + P22 * phi0
    scale = -1.0 / ms2

    out[0] = dtheta
    out[1] = domega
    out[2] = di_a
    out[3] = de_int
    out[4] = dg_z
    out[5] = dh_z
    out[6] = dg_theta
    out[7] = dh_theta
    out[8] = dparam0
    out[9] = dparam1
    out[10] = dparam2
    out[11] = scale * v0 * v0
    out[12] = scale * v0 * v1
    out[13] = scale * v0 * v2
    out[14] = scale * v1 * v0
    out[15] = scale * v1 * v1
    out[16] = scale * v1 * v2
    out[17] = scale * v2 * v0
    out[18] = scale * v2 * v1
    out[19] = scale * v2 * v2

    return out

@njit(cache=True)
def rk4_step_nb(t, x, h):
    k1 = rhs_nb(t, x)
    k2 = rhs_nb(t + 0.5 * h, x + 0.5 * h * k1)
    k3 = rhs_nb(t + 0.5 * h, x + 0.5 * h * k2)
    k4 = rhs_nb(t + h, x + h * k3)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

@njit(cache=True)
def simulate_fast_nb(t_eval, dt_max):
    n = len(t_eval)
    y = np.zeros((20, n))

    x = np.zeros(20)
    x[8] = M_est_0
    x[9] = D_est_0
    x[10] = K_est_0
    x[11:20] = P0.reshape(9)

    y[:, 0] = x
    t = t_eval[0]

    for i in range(1, n):
        t_target = t_eval[i]

        while t < t_target:
            h = dt_max
            remaining = t_target - t
            if remaining < h:
                h = remaining
            x = rk4_step_nb(t, x, h)
            t = t + h

        y[:, i] = x

    return y


def run_simulation():
    x0 = [
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        M_est_0, D_est_0, K_est_0,
        *P0.reshape(-1)
    ]

    if USE_FAST_SIM and NUMBA_AVAILABLE:
        y_eval = simulate_fast_nb(t_eval, FAST_DT)
        method_used = f"numba RK4 exact-t_eval, dt_max={FAST_DT}"
        return t_eval, y_eval, method_used

    sol = solve_ivp(
        dynamics,
        [0, T],
        x0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-6,
        atol=1e-9,
        max_step=2e-3,
    )
    method_used = "solve_ivp LSODA"
    return t_eval, sol.y, method_used


if __name__ == "__main__":
    t_eval, y_eval, method_used = run_simulation()

    theta_sol = y_eval[0]
    omega_sol = y_eval[1]
    i_a_sol = y_eval[2]
    e_int_sol = y_eval[3]

    M_hat_sol = y_eval[8]
    D_hat_sol = y_eval[9]
    K_hat_sol = y_eval[10]

    g_z_sol = y_eval[4]
    h_z_sol = y_eval[5]
    g_th_sol = y_eval[6]
    h_th_sol = y_eval[7]

    z_vec = h_z_sol
    phi_0 = h_th_sol
    phi_1 = g_th_sol - lambda1 * h_th_sol
    phi_2 = theta_sol - (lambda0 + lambda1) * phi_1 - lambda0 * lambda1 * phi_0

    Phi_mat = np.vstack([phi_2, phi_1, phi_0]).T
    theta_ls = np.linalg.lstsq(Phi_mat, z_vec, rcond=None)[0]

    print(f"Simulation method: {method_used}")
    print("Final adaptive estimate:")
    print(f"  M_hat = {M_hat_sol[-1]:.6f}   (true {M_true:.6f})")
    print(f"  D_hat = {D_hat_sol[-1]:.6f}   (true {D_true:.6f})")
    print(f"  K_hat = {K_hat_sol[-1]:.6f}   (true {K_true:.6f})")

    print("\nOffline LS on the same filtered data:")
    print(f"  M_ls  = {theta_ls[0]:.6f}")
    print(f"  D_ls  = {theta_ls[1]:.6f}")
    print(f"  K_ls  = {theta_ls[2]:.6f}")

    V_ideal = control_law(theta_sol, r(t_eval) - theta_sol, omega_sol, e_int_sol)

    t_pwm = np.linspace(0, T, 20 * int(T * pwm_frequency))
    V_ideal_dense = np.interp(t_pwm, t_eval, V_ideal)
    pwm_signal = pwm_voltage(t_pwm, V_ideal_dense)

    samples_per_period = 20
    kernel = np.ones(samples_per_period) / samples_per_period
    pwm_average = np.convolve(pwm_signal, kernel, mode="same")

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
    if SAVE_PLOT:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f"{save_folder}/{filename}", dpi=150)

    if SHOW_PLOT:
        plt.show()
