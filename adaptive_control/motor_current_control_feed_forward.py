
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern} \usepackage[T1]{fontenc}",
})

"""
Simulation parameters
"""
T  = 6*10**-3       # Total simulation time (s)
fs = 100_000   # Sample rate (Hz)  – fixed step, fully transparent
dt = 1.0 / fs
N  = int(T * fs) + 1
t_eval = np.linspace(0, T, N)

should_disable_feed_forward = False
should_disable_noise = True
should_show_plot = False
should_cover_whole_page = False

t0 = 2*10**-3
t1 = 3*10**-3
r_max = 1.0

fc = 5_000   # Controller update rate (Hz)
dt_c = 1.0 / fc
print(f"Controller update period: {dt_c*1000:.2f} ms")

theta_0 = 0.0  # Initial angle (rad)
omega_0 = 0.0  # Initial angular velocity (rad/s)
i_a_0 = 0.0    # Initial armature current (A)

filename = f"motor_current_control_feed_forward_P{'_FF' if not should_disable_feed_forward else ''}{'_noise' if not should_disable_noise else ''}_Ts_{dt_c*1000:.1f}ms"
base_folder = r"C:\Users\Anders\OneDrive - NTNU\Fordypningsoppgave - Nsquared\specialization_project\Thesis\Figures"  # Laptop
base_folder = r"C:\Users\ahe02\OneDrive - NTNU\Fordypningsoppgave - Nsquared\specialization_project\Thesis\Figures"  # Stasjonær-PC

"""
Define motor parameters
"""

Jm = 5.0e-4
Bm = 6.47e-3   # N·m·s (Rotor friction coefficient)
Kb = 0.845     # V·s/rad (Back EMF constant)
Kt = 0.44      # N·m/A (Torque constant)
Ra = 2.44      # Ω (Armature resistance)
La = 0.006     # H (Armature inductance)

E_a_sat = 12.0   # V (Supply voltage / saturation limit)
pwm_frequency = 1000.0  # Hz (PWM frequency)
pwm_period = 1.0 / pwm_frequency  # s (PWM period)

scale = 0.5
exp_sol = np.exp(-Ra / fc)

Kp = La * Ra * (1 + exp_sol) / (1 - exp_sol) * scale

print(f"Kp = {Kp:.2f}")


# Measurement noise standard deviations (set to 0 to disable)
noise_std_Ia    = 0.05  # A
if should_disable_noise:
    noise_std_Ia    =0.0

f_ref = 50.0  # Hz (sinusoidal reference frequency)
r = lambda t: r_max * ((t > 1/2*t0) & (t < t1)) - r_max * (t >= t1)  # Step reference input
r = lambda t: r_max * np.sin(2 * np.pi * f_ref * t)                   # Sinusoidal reference
r = lambda t: r_max * (t >= t0)                                         # Step reference input


"""
System dynamics

Transfer function:  θ/V = Kt / (Jm*La*s³ + (Jm*Ra + Bm*La)*s² + (Ra*Bm + Kt*Kb)*s)
State-space realisation  x = [θ, ω, i_a]:
  dθ/dt   = ω
  dω/dt   = (Kt*i_a - Bm*ω) / Jm
  di_a/dt = (V - Ra*i_a - Kb*ω) / La
"""


def voltage_clamp(V):
    return np.clip(V, -E_a_sat, E_a_sat)


def control_law(i_a, omega, t):
    r_t = r(t) * (Ra + Kp) / Kp
    V = Kp * (r_t - i_a)
    if not should_disable_feed_forward:
        V += Kb * omega
    return voltage_clamp(V)


def dynamics(x, V):
    """State derivatives given state x = [theta, omega, i_a] and applied voltage V."""
    theta, omega, i_a = x
    dtheta = omega
    domega = (Kt * i_a - Bm * omega) / Jm
    di_a   = (V - Ra * i_a - Kb * omega) / La
    return np.array([dtheta, domega, di_a])


def rk4_step(x, V, dt):
    """Single RK4 step with voltage V held constant over the step."""
    k1 = dynamics(x,              V)
    k2 = dynamics(x + dt/2 * k1,  V)
    k3 = dynamics(x + dt/2 * k2,  V)
    k4 = dynamics(x + dt   * k3,  V)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


if __name__ == "__main__":
    # True (noiseless) states
    theta_arr      = np.zeros(N)
    omega_arr      = np.zeros(N)
    i_a_arr        = np.zeros(N)
    # Noisy measurements seen by the controller
    omega_meas_arr = np.zeros(N)
    omega_fd_arr   = np.zeros(N)  # clean finite-difference estimate (no noise)
    i_a_meas_arr   = np.zeros(N)
    E_a_desired   = np.zeros(N)
    E_a_applied   = np.zeros(N)

    x          = np.array([theta_0, omega_0, i_a_0])
    rng        = np.random.default_rng(seed=1)
    # Initialise so that the first finite-difference estimate reproduces omega_0
    theta_meas_prev = theta_0 - omega_0 * dt_c
    theta_prev      = theta_0 - omega_0 * dt_c  # clean (noiseless) FD tracker

    # Controller state – held constant between controller updates
    next_ctrl_t = 0.0          # time of next allowed controller update
    E_a_des  = 0.0
    E_a_app  = 0.0
    omega_meas = omega_0
    omega_fd   = omega_0

    for i in tqdm(range(N)):
        t = t_eval[i]
        theta_true, omega_true, i_a_true = x

        # Controller fires only at fc Hz; measurements are taken and voltage updated at that rate
        if t >= next_ctrl_t - 1e-12:
            # Noisy measurements - only sampled at controller rate
            i_a_meas        = i_a_true  + rng.normal(0.0, noise_std_Ia)
            theta_meas      = theta_true
            omega_meas      = (theta_meas - theta_meas_prev) / dt_c
            theta_meas_prev = theta_meas
            # Clean finite-difference estimate (no noise on theta)
            omega_fd        = (theta_true - theta_prev) / dt_c
            theta_prev      = theta_true

            E_a_des  = control_law(i_a_meas, omega_meas, t)
            E_a_app  = E_a_des
            next_ctrl_t += dt_c

        # Save true states
        theta_arr[i] = x[0]
        omega_arr[i] = omega_true
        i_a_arr[i]   = i_a_true
        # Save noisy measurements and voltages
        omega_meas_arr[i] = omega_meas
        omega_fd_arr[i]   = omega_fd
        i_a_meas_arr[i]   = i_a_meas
        E_a_desired[i]  = E_a_des
        E_a_applied[i]  = E_a_app

        if i < N - 1:
            x = rk4_step(x, E_a_app, dt)

    r_vals = r(t_eval)

    di_a_dt = np.diff(i_a_arr) / dt
    print(f"Max rate of change in true current: {np.max(np.abs(di_a_dt)):.2f} A/s")

    t_eval = t_eval * 1000
    if should_cover_whole_page:
        TITLE_SIZE = 30
        LABEL_SIZE = 22
        TICK_SIZE  = 22
        LEGEND_SIZE = 20
        LINE_WIDTH = 3
        LINE_NOISE_WIDTH = 0.8
    else:
        TITLE_SIZE  = 42
        LABEL_SIZE  = 36
        TICK_SIZE   = 34
        LEGEND_SIZE = 28
        LINE_WIDTH  = 4
        LINE_NOISE_WIDTH = 1.5
    LINE_NOISE_WIDTH = LINE_WIDTH

    fig, axes = plt.subplots(3, 1, figsize=(12, int(10/4*5.5)), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1.5]})
    fig.suptitle(f'Current control - P{" + FF" if not should_disable_feed_forward else ""}{" - with noise" if not should_disable_noise else ""}', fontsize=TITLE_SIZE)

    axes[0].plot(t_eval, r_vals, 'r--', linewidth=0.7*LINE_WIDTH, label=r'$r(t)$')
    axes[0].plot(t_eval, i_a_meas_arr, color='tab:orange', linewidth=LINE_NOISE_WIDTH, label=r'$I_a[k]$' + r" $ + \; w_{I_a}[k]$" * (int(not should_disable_noise)))


    axes[0].plot(t_eval, i_a_arr, color='tab:blue',     linewidth=LINE_WIDTH, label=r'$I_a(t)$')
    axes[0].set_ylabel('Current [A]', fontsize=LABEL_SIZE)
    axes[0].tick_params(axis='both', labelsize=TICK_SIZE)
    axes[0].yaxis.set_major_locator(plt.MultipleLocator(1))
    axes[0].legend(fontsize=LEGEND_SIZE, loc='lower right')
    axes[0].grid()

    # axes[1].plot(t_eval, V_desired, '--', label=r'$V_\mathrm{desired}$ (unclamped)')
    axes[1].axhline( E_a_sat, color='red', linestyle='--', linewidth=0.7*LINE_WIDTH, label=r'$E_{a,\mathrm{sat}}=\pm12V$')
    axes[1].axhline(-E_a_sat, color='red', linestyle='--', linewidth=0.7*LINE_WIDTH)
    axes[1].plot(t_eval, E_a_applied,  color='tab:orange', linewidth=LINE_WIDTH, label=r'$E_a(t)$')
    axes[1].set_ylabel(r'Voltage [V]', fontsize=LABEL_SIZE)
    axes[1].tick_params(axis='both', labelsize=TICK_SIZE)
    axes[1].legend(fontsize=LEGEND_SIZE, loc='lower right')
    axes[1].grid()

    if not should_disable_feed_forward:
        axes[2].plot(t_eval, omega_fd_arr, color='C1', linewidth=LINE_WIDTH, label=r'$\hat{\omega}[k]$')
    axes[2].plot(t_eval, omega_arr, color='C0', linewidth=LINE_WIDTH, label=r'$\omega(t)$')
    axes[2].set_xlabel('Time [ms]', fontsize=LABEL_SIZE)
    axes[2].set_ylabel(r'Ang. vel. [rad/s]', fontsize=LABEL_SIZE)
    axes[2].tick_params(axis='both', labelsize=TICK_SIZE)
    axes[2].legend(fontsize=LEGEND_SIZE, loc='lower right')
    axes[2].grid()

    plt.tight_layout()
    plt.savefig(fr'adaptive_control\figures\{filename}.pdf', bbox_inches='tight')
    plt.savefig(fr'{base_folder}\Controller_design\{filename}.pdf', bbox_inches='tight')
    if should_show_plot:
        plt.show()