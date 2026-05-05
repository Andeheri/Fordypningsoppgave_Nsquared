import numpy as np
from numpy import pi
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
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

Jm = 0.093   # kg·m²  (true rotor inertia  – unknown to Jm estimator)
La = 0.006   # H      (true armature inductance – unknown to La estimator)

# ─── PI-PD controller ────────────────────────────────────────────────────────
Kp1, Ki, Kp2, Kd = 50.0, 3.0, 2.0, 5.0
t0, r_max = 0.05, 5.0
r = lambda t: r_max * (t > t0)   # step reference [rad]

# ─── Inertia estimator (second-order filter Λ = (s+λ0)(s+λ1)) ───────────────
# Mechanical equation filtered by 1/Λ:
#   z  = 1/Λ [Kt·Ia]  −  Bm·s/Λ [θ]
#   Φ  = s²/Λ [θ]
#   SPM: z = Jm · Φ
#   dJm_est/dt = Γ_Jm · ε_Jm · Φ_Jm,   ε_Jm = z − Jm_est · Φ_Jm

lambda0_jm  = 0.1
lambda1_jm  = 0.1
Gamma_jm    = 1000.0
Jm_est_0    = 0.01    # initial estimate [kg·m²]

# ─── Inductance estimator (first-order filter γ = s+λ0) ─────────────────────
# Electrical equation filtered by 1/γ:
#   z  = 1/γ [V] − 1/γ [Ra·Ia] − s/γ [Kb·θ]
#   Φ  = s/γ [Ia]
#   SPM: z = La · Φ
#   dLa_est/dt = Γ_La · ε_La · Φ_La,   ε_La = z − La_est · Φ_La

lambda0_la  = 10.0
Gamma_la    = 10.0
La_est_0    = 0.001   # initial estimate [H]

# ─── Simulation parameters ────────────────────────────────────────────────────
T      = 0.3
t_eval = np.linspace(0, T, 10_000)

save_folder = "adaptive_control/figures"
filename    = "inertia_and_inductance_estimator_gradient_results.png"

# ─── Measurement noise ────────────────────────────────────────────────────────
MEASUREMENT_NOISE = False    # toggle on/off
noise_std_theta   = 0.01    # rad  – angle measurement noise (std dev)
noise_std_i_a     = 0.05    # A    – current measurement noise (std dev)


# ─── Noise interpolators (set in __main__) ───────────────────────────────────
_noise_theta_fn = None
_noise_i_a_fn   = None

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
# State vector:
#   x[0]  theta_angle
#   x[1]  omega
#   x[2]  i_a
#   x[3]  e_int
#   ── Jm estimator (2nd-order filter) ──
#   x[4]  f_ktia_0    1/(s+λ0_jm) [Kt·Ia]
#   x[5]  f_ktia_1    1/Λ         [Kt·Ia]
#   x[6]  f_theta_0   1/(s+λ0_jm) [θ]
#   x[7]  f_theta_1   1/Λ         [θ]
#   x[8]  Jm_est
#   ── La estimator (1st-order filter) ──
#   x[9]  f_la1       1/γ [V]
#   x[10] f_la2       1/γ [Ra·Ia]
#   x[11] f_la3       1/γ [Kb·θ]
#   x[12] f_la4       1/γ [Ia]
#   x[13] La_est

def dynamics(t, x):
    theta_angle, omega, i_a, e_int = x[0], x[1], x[2], x[3]
    f_ktia_0, f_ktia_1             = x[4], x[5]
    f_theta_0, f_theta_1           = x[6], x[7]
    Jm_est                         = x[8]
    f_la1, f_la2, f_la3, f_la4     = x[9], x[10], x[11], x[12]
    La_est                         = x[13]

    # Control
    theta_err = r(t) - theta_angle
    V_ideal   = control_law(theta_angle, theta_err, omega, e_int)
    V         = pwm_voltage(t, V_ideal)

    # Motor dynamics
    dtheta = omega
    domega = (Kt * i_a - Bm * omega) / Jm
    di_a   = (V - Ra * i_a - Kb * omega) / La
    de_int = theta_err

    # ── Noisy measurements (used by estimators only) ───────────────────────────
    if MEASUREMENT_NOISE and _noise_theta_fn is not None:
        theta_meas = theta_angle + _noise_theta_fn(t)
        i_a_meas   = i_a         + _noise_i_a_fn(t)
    else:
        theta_meas = theta_angle
        i_a_meas   = i_a

    # ── Jm estimator filters (second-order Λ = (s+λ0_jm)(s+λ1_jm)) ──────────
    df_ktia_0 = -lambda0_jm * f_ktia_0 + Kt * i_a_meas
    df_ktia_1 = -lambda1_jm * f_ktia_1 + f_ktia_0

    df_theta_0 = -lambda0_jm * f_theta_0 + theta_meas
    df_theta_1 = -lambda1_jm * f_theta_1 + f_theta_0

    s_over_Lambda_theta  = f_theta_0 - lambda1_jm * f_theta_1
    s2_over_Lambda_theta = (
        theta_meas
        - (lambda0_jm + lambda1_jm) * s_over_Lambda_theta
        - lambda0_jm * lambda1_jm * f_theta_1
    )

    z_jm   = f_ktia_1 - Bm * s_over_Lambda_theta
    Phi_jm = s2_over_Lambda_theta

    eps_jm  = z_jm - Jm_est * Phi_jm
    dJm_est = Gamma_jm * eps_jm * Phi_jm

    # ── La estimator filters (first-order γ = s+λ0_la) ───────────────────────
    df_la1 = -lambda0_la * f_la1 + V
    df_la2 = -lambda0_la * f_la2 + Ra * i_a_meas
    df_la3 = -lambda0_la * f_la3 + Kb * theta_meas
    df_la4 = -lambda0_la * f_la4 + i_a_meas

    z_la   = f_la1 - f_la2 - (Kb * theta_meas - lambda0_la * f_la3)
    Phi_la = i_a_meas - lambda0_la * f_la4

    eps_la  = z_la - La_est * Phi_la
    dLa_est = Gamma_la * eps_la * Phi_la

    return [
        dtheta, domega, di_a, de_int,
        df_ktia_0, df_ktia_1,
        df_theta_0, df_theta_1,
        dJm_est,
        df_la1, df_la2, df_la3, df_la4,
        dLa_est,
    ]


if __name__ == "__main__":
    # ── Set up noise interpolators ────────────────────────────────────────────
    if MEASUREMENT_NOISE:
        rng = np.random.default_rng(seed=42)
        _noise_theta_fn = interp1d(
            t_eval, rng.normal(0, noise_std_theta, len(t_eval)),
            kind='linear', fill_value='extrapolate'
        )
        _noise_i_a_fn = interp1d(
            t_eval, rng.normal(0, noise_std_i_a, len(t_eval)),
            kind='linear', fill_value='extrapolate'
        )

    x0 = [
        0.0, 0.0, 0.0, 0.0,     # theta_angle, omega, i_a, e_int
        0.0, 0.0,               # f_ktia_0, f_ktia_1
        0.0, 0.0,               # f_theta_0, f_theta_1
        Jm_est_0,               # Jm_est
        0.0, 0.0, 0.0, 0.0,     # f_la1, f_la2, f_la3, f_la4
        La_est_0,               # La_est
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

    theta_sol  = sol.y[0]
    omega_sol  = sol.y[1]
    i_a_sol    = sol.y[2]
    e_int_sol  = sol.y[3]
    Jm_est_sol = sol.y[8]
    La_est_sol = sol.y[13]

    # Compute voltage signals for plotting
    V_ideal = control_law(theta_sol, r(t_eval) - theta_sol, omega_sol, e_int_sol)
    t_ms = t_eval * 1e3   # convert to milliseconds for plotting

    # ─── Plot style parameters ────────────────────────────────────────────────
    TITLE_SIZE  = 30
    LABEL_SIZE  = 22
    TICK_SIZE   = 22
    LEGEND_SIZE = 20
    LINE_WIDTH  = 3

    # ─── Plots ────────────────────────────────────────────────────────────────
    def setup_axis(ax, ylabel, color='black', title=None, show_xlabel=False):
        if title:
            ax.set_title(title, fontsize=TITLE_SIZE, pad=15)
        if show_xlabel:
            ax.set_xlabel('Time [ms]', fontsize=LABEL_SIZE)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False)
        ax.set_ylabel(ylabel, color=color, fontsize=LABEL_SIZE)
        ax.tick_params(axis='y', labelcolor=color, labelsize=TICK_SIZE)
        ax.tick_params(axis='x', labelsize=TICK_SIZE)
        ax.grid()

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

    # ── Subplot 0: both estimates on twin axes ────────────────────────────────
    ax_la = axes[0]
    ax_jm = ax_la.twinx()

    ax_la.axhline(La * 1e3, color='tab:red', linestyle='--', label=f'$L_a$ = {La*1e3:.1f} mH\n$J_m$ = {Jm*1e3:.1f} ' + r'mkg·m²', linewidth=LINE_WIDTH)
    ax_la.plot(t_ms, La_est_sol * 1e3,  color='tab:orange', label=r'$\hat{L}_a(t)$',  linewidth=LINE_WIDTH)
    ax_jm.plot(t_ms, Jm_est_sol * 1e3,  color='tab:blue',   label=r'$\hat{J}_m(t)$',  linewidth=LINE_WIDTH)
    # ax_jm.axhline(Jm, color='tab:blue',   linestyle='--',     label=rf'$J_m$ = {Jm:.4f} kg·m²', linewidth=LINE_WIDTH)

    lines_jm, labels_jm = ax_jm.get_legend_handles_labels()
    lines_la, labels_la = ax_la.get_legend_handles_labels()
    ax_jm.legend(lines_la + lines_jm, labels_la + labels_jm, fontsize=LEGEND_SIZE, loc='lower right')

    setup_axis(ax_jm, r'Inertia [mkg·m$^2$]', color='tab:blue', title='Rotor Inertia & Armature Inductance Estimates')
    ax_jm.grid(False)
    ax_la.grid(True)
    ax_la.set_ylabel('Inductance [mH]', color='tab:orange', fontsize=LABEL_SIZE)
    ax_la.tick_params(axis='y', labelcolor='tab:orange', labelsize=TICK_SIZE)
    ax_la.tick_params(axis='x', labelbottom=False)

    # ── Subplot 1: angle ──────────────────────────────────────────────────────
    axes[1].plot(t_ms, theta_sol, color='tab:blue', label=r'$\theta(t)$', linewidth=LINE_WIDTH)
    axes[1].legend(fontsize=LEGEND_SIZE, loc='lower right')
    setup_axis(axes[1], 'Angle [rad]')

    # ── Subplot 2: current (right) + voltage (left), zero-aligned ────────────
    ax_v = axes[2]
    ax_i = ax_v.twinx()

    ax_v.plot(t_ms, V_ideal, color='tab:orange', label=r'$E_a(t)$',    linewidth=LINE_WIDTH)
    ax_i.plot(t_ms, i_a_sol, color='tab:blue',   label=r'$I_a(t)$', linewidth=LINE_WIDTH)

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
    ax_v.legend(lines_v + lines_i, labels_v + labels_i, loc='lower right', fontsize=LEGEND_SIZE)
    setup_axis(ax_v, 'Voltage [V]', color='tab:orange', show_xlabel=True)
    ax_i.set_ylabel('Current [A]', color='tab:blue', fontsize=LABEL_SIZE)
    ax_i.tick_params(axis='y', labelcolor='tab:blue', labelsize=TICK_SIZE)

    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(f"{save_folder}/{filename}", dpi=150)
    plt.savefig(fr"C:\Users\Anders\OneDrive - NTNU\Fordypningsoppgave - Nsquared\specialization_project\Thesis\Figures\Modelling\{filename}", dpi=150)
    