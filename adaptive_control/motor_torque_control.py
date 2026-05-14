
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from motor_parameters import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern} \usepackage[T1]{fontenc}",
})

"""
Simulation parameters
"""
T      = 10e-3    # Total simulation time (s)
t_wall = 5e-3    # Time at which motor hits the wall (s)
fs     = 100_000  # Sample rate (Hz) – fixed step
dt     = 1.0 / fs
N      = int(T * fs) + 1
t_eval = np.linspace(0, T, N)

should_disable_noise    = True
should_show_plot        = False
should_cover_whole_page = False

V_spinup = 3.0   # V   – open-loop voltage during free-spin phase
r_tau    = 1.5   # N·m – reference torque to exert on the wall

fc   = 5_000         # Controller update rate (Hz)
dt_c = 1.0 / fc
print(f"Controller update period: {dt_c*1000:.2f} ms")

theta_0 = 0.0  # Initial angle (rad)
omega_0 = 0.0  # Initial angular velocity (rad/s)
i_a_0   = 0.0  # Initial armature current (A)

filename    = f"motor_torque_control_P{'_noise' if not should_disable_noise else ''}_Ts_{dt_c*1000:.1f}ms"
base_folder = r"C:\Users\Anders\OneDrive - NTNU\Fordypningsoppgave - Nsquared\specialization_project\Thesis\Figures"  # Laptop
base_folder = r"C:\Users\ahe02\OneDrive - NTNU\Fordypningsoppgave - Nsquared\specialization_project\Thesis\Figures"  # Stasjonær-PC

"""
Define motor parameters
"""

E_a_sat = 12.0  # V (Supply voltage / saturation limit)

scale   = 0.5
exp_sol = np.exp(-Ra / fc)
Kp      = La * Ra * (1 + exp_sol) / (1 - exp_sol) * scale
print(f"Kp = {Kp:.4f}")

# Measurement noise standard deviations (set to 0 to disable)
noise_std_Ia = 0.05  # A
if should_disable_noise:
    noise_std_Ia = 0.0

"""
System dynamics  x = [θ, ω, i_a]
  dθ/dt   = ω
  dω/dt   = (Kt·i_a - Bm·ω) / Jm   (0 when at wall – position locked)
  di_a/dt = (V - Ra·i_a - Kb·ω) / La
"""


def voltage_clamp(V):
    return np.clip(V, -E_a_sat, E_a_sat)


def control_law(i_a, t):
    if t < t_wall:
        # Free-spin phase: open-loop spin-up
        return voltage_clamp(V_spinup)
    # Wall phase: P current controller, reference current = r_tau / Kt
    r_I_scaled = (r_tau / Kt) * (Ra + Kp) / Kp
    return voltage_clamp(Kp * (r_I_scaled - i_a))


def dynamics(x, V, at_wall=False):
    theta, omega, i_a = x
    if at_wall:
        omega = 0.0
    dtheta = omega
    domega = 0.0 if at_wall else (Kt * i_a - Bm * omega) / Jm
    di_a   = (V - Ra * i_a - Kb * omega) / La
    return np.array([dtheta, domega, di_a])


def rk4_step(x, V, dt, at_wall=False):
    k1 = dynamics(x,              V, at_wall)
    k2 = dynamics(x + dt/2 * k1,  V, at_wall)
    k3 = dynamics(x + dt/2 * k2,  V, at_wall)
    k4 = dynamics(x + dt   * k3,  V, at_wall)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


if __name__ == "__main__":
    theta_arr    = np.zeros(N)
    omega_arr    = np.zeros(N)
    i_a_arr      = np.zeros(N)
    tau_arr      = np.zeros(N)
    i_a_meas_arr = np.zeros(N)
    E_a_applied  = np.zeros(N)

    x   = np.array([theta_0, omega_0, i_a_0])
    rng = np.random.default_rng(seed=1)

    theta_wall  = None  # position locked at first wall contact
    next_ctrl_t = 0.0
    E_a_app     = 0.0
    i_a_meas    = i_a_0

    for i in tqdm(range(N)):
        t = t_eval[i]
        theta_true, omega_true, i_a_true = x
        at_wall = t >= t_wall

        if at_wall and theta_wall is None:
            theta_wall = theta_true  # lock position on first contact

        # Controller fires at fc Hz
        if t >= next_ctrl_t - 1e-12:
            i_a_meas = i_a_true + rng.normal(0.0, noise_std_Ia)
            E_a_app  = control_law(i_a_meas, t)
            next_ctrl_t += dt_c

        theta_arr[i]    = theta_true
        omega_arr[i]    = omega_true
        i_a_arr[i]      = i_a_true
        tau_arr[i]      = Kt * i_a_true
        i_a_meas_arr[i] = i_a_meas
        E_a_applied[i]  = E_a_app

        if i < N - 1:
            x = rk4_step(x, E_a_app, dt, at_wall)
            if at_wall:
                x[0] = theta_wall  # enforce locked position
                x[1] = 0.0         # enforce zero velocity

    tau_ref_vals  = np.where(t_eval >= t_wall, r_tau, np.nan)
    tau_meas_wall = np.where(t_eval >= t_wall, Kt * i_a_meas_arr, np.nan)
    t_eval_ms    = t_eval * 1000
    t_wall_ms    = t_wall * 1000

    if should_cover_whole_page:
        TITLE_SIZE  = 30
        LABEL_SIZE  = 22
        TICK_SIZE   = 22
        LEGEND_SIZE = 20
        LINE_WIDTH  = 3
    else:
        TITLE_SIZE  = 42
        LABEL_SIZE  = 36
        TICK_SIZE   = 34
        LEGEND_SIZE = 24
        LINE_WIDTH  = 4

    fig, axes = plt.subplots(3, 1, figsize=(12, int(10/4*5.5)), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1.5]})
    fig.suptitle(r'Torque control -- P Controller', fontsize=TITLE_SIZE)

    for ax in axes:
        ax.axvline(t_wall_ms, color='gray', linestyle=':', linewidth=0.8*LINE_WIDTH, label=r'Wall contact')

    axes[0].plot(t_eval_ms, tau_ref_vals, 'r--', linewidth=0.7*LINE_WIDTH, label=r'$\tau_\mathrm{ref}$')
    axes[0].plot(t_eval_ms, tau_meas_wall, color='tab:orange', linewidth=LINE_WIDTH, label=r'$\tau[k] = K_t I_a[k]$')
    axes[0].plot(t_eval_ms, tau_arr, color='tab:blue', linewidth=LINE_WIDTH, label=r'$\tau(t) = K_t I_a(t)$')
    axes[0].set_ylabel(r'Torque [N$\cdot$m]', fontsize=LABEL_SIZE)
    axes[0].tick_params(axis='both', labelsize=TICK_SIZE)
    axes[0].legend(fontsize=LEGEND_SIZE, loc='lower right')
    axes[0].grid()
    ax0_current = axes[0].secondary_yaxis('right', functions=(lambda tau: tau / Kt, lambda I: I * Kt))
    ax0_current.set_ylabel(r'Current [A]', fontsize=LABEL_SIZE)
    ax0_current.tick_params(axis='y', labelsize=TICK_SIZE)

    axes[1].axhline( E_a_sat, color='red', linestyle='--', linewidth=0.7*LINE_WIDTH, label=r'$E_{a,\mathrm{sat}}=\pm12\,\mathrm{V}$')
    axes[1].axhline(-E_a_sat, color='red', linestyle='--', linewidth=0.7*LINE_WIDTH)
    axes[1].plot(t_eval_ms, E_a_applied, color='tab:orange', linewidth=LINE_WIDTH, label=r'$E_a(t)$')
    axes[1].set_ylabel(r'Voltage [V]', fontsize=LABEL_SIZE)
    axes[1].tick_params(axis='both', labelsize=TICK_SIZE)
    axes[1].legend(fontsize=LEGEND_SIZE, loc='lower right')
    axes[1].grid()

    axes[2].plot(t_eval_ms, omega_arr, color='C0', linewidth=LINE_WIDTH, label=r'$\omega(t)$')
    axes[2].set_xlabel('Time [ms]', fontsize=LABEL_SIZE)
    axes[2].set_ylabel(r'Ang. vel. [rad/s]', fontsize=LABEL_SIZE)
    axes[2].tick_params(axis='both', labelsize=TICK_SIZE)
    axes[2].legend(fontsize=LEGEND_SIZE, loc='upper right')
    axes[2].grid()

    plt.tight_layout()
    plt.savefig(fr'adaptive_control\figures\{filename}.pdf', bbox_inches='tight')
    plt.savefig(fr'{base_folder}\Controller_design\{filename}.pdf', bbox_inches='tight')
    if should_show_plot:
        plt.show()