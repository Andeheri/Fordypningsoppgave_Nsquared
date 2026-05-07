
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Latin Modern Roman"],
#     "text.latex.preamble": r"\usepackage{lmodern} \usepackage[T1]{fontenc}",
# })

"""
Simulation parameters
"""
T  = 8*10**-3       # Total simulation time (s)
fs = 100_000   # Sample rate (Hz)  – fixed step, fully transparent
dt = 1.0 / fs
N  = int(T * fs) + 1
t_eval = np.linspace(0, T, N)

fc = 10_000     # Controller update rate (Hz)
dt_c = 1.0 / fc

theta_0 = 0.0  # Initial angle (rad)
omega_0 = 5.0  # Initial angular velocity (rad/s)
i_a_0 = 0.0    # Initial armature current (A)

"""
Define motor parameters
"""

Jm = 0.093  # kg*m^2 (Rotor moment of inertia)
Bm = 0.008  # N*m*s (Rotor friction coefficient)
Kb = 0.6    # V*s/rad (Back EMF constant)
Kt = 0.7274 # Nm/A (Torque constant)
Ra = 0.6    # Ω (Armature resistance)
La = 0.006  # H (Armature inductance)

E_a_sat = 12.0   # V (Supply voltage / saturation limit)
pwm_frequency = 1000.0  # Hz (PWM frequency)
pwm_period = 1.0 / pwm_frequency  # s (PWM period)

"""
PI-PD control parameters

Control law (2-DOF structure):
  V = Kp1*(r - θ) + Ki*∫(r - θ)dt  [PI on error]
    - Kp2*ω - Kd*(Kt*i_a - Bm*ω)/Jm [PD feedback on output]
"""

Kp = 90
Ki = 0

print(f"Kp = {Kp:.2f}, Ki = {Ki:.2f}")
should_disable_feed_forward = False
should_disable_noise = False
should_disable_filter = True

t0 = 2*10**-3
t1 = 3*10**-3
r_max = 2.0

# Measurement noise standard deviations (set to 0 to disable)
noise_std_Ia    = 0.05   # A
noise_std_omega = 0.008    # rad/s
if should_disable_noise:
    noise_std_Ia = 0.0
    noise_std_omega = 0.0

# ─── One Euro Filter parameters (applied to control voltage) ─────────────────
oef_min_cutoff = 100.0   # Hz  – lower → smoother but more lag
oef_beta       = 20.0     # Hz/(V/s) – higher → less lag during fast changes
oef_dcutoff    = 1.0     # Hz  – derivative low-pass cutoff

r = lambda t: r_max * ((t > t0) & (t < t1)) - r_max * (t >= t1)  # Step reference input
r = lambda t: r_max * (t > t0) 

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


def control_law(i_a, omega, e_int, t):
    r_t = r(t) * (2 - Kp / (Ra + Kp))
    V = 1 * (Kp * (r_t - i_a) + Ki * e_int)
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


class OneEuroFilter:
    """Adaptive low-pass filter (Casiez et al., 2012)."""
    def __init__(self, dt, min_cutoff=1.0, beta=0.0, dcutoff=1.0):
        self.dt = dt
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev = None
        self._dx_hat = 0.0

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / self.dt)

    def __call__(self, x):
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) / self.dt
        alpha_d = self._alpha(self.dcutoff)
        self._dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_hat
        cutoff = self.min_cutoff + self.beta * abs(self._dx_hat)
        alpha = self._alpha(cutoff)
        x_hat = alpha * x + (1.0 - alpha) * self._x_prev
        self._x_prev = x_hat
        return x_hat


if __name__ == "__main__":
    # True (noiseless) states
    theta_arr      = np.zeros(N)
    omega_arr      = np.zeros(N)
    i_a_arr        = np.zeros(N)
    e_int_arr      = np.zeros(N)
    # Noisy measurements seen by the controller
    omega_meas_arr = np.zeros(N)
    i_a_meas_arr   = np.zeros(N)
    E_a_desired   = np.zeros(N)   # unclamped controller output
    E_a_applied   = np.zeros(N)   # clamped, before filter
    E_a_filtered  = np.zeros(N)   # after One Euro Filter

    x          = np.array([theta_0, omega_0, i_a_0])
    e_int      = 0.0
    rng        = np.random.default_rng(seed=1)
    oef_V      = OneEuroFilter(dt_c, min_cutoff=oef_min_cutoff, beta=oef_beta, dcutoff=oef_dcutoff)

    # Controller state – held constant between controller updates
    next_ctrl_t = 0.0          # time of next allowed controller update
    E_a_des  = 0.0
    E_a_app  = 0.0
    E_a_filt = 0.0

    for i in tqdm(range(N)):
        t = t_eval[i]
        _, omega_true, i_a_true = x

        # Noisy measurements - controller only sees these
        i_a_meas   = i_a_true   + rng.normal(0.0, noise_std_Ia)
        omega_meas = omega_true + rng.normal(0.0, noise_std_omega)

        # Controller fires only at fc Hz; voltage is held between updates
        if t >= next_ctrl_t - 1e-12:
            E_a_des  = control_law(i_a_meas, omega_meas, e_int, t)  # unclamped control output
            E_a_app  = voltage_clamp(E_a_des)          # clamped, before filter
            E_a_filt = E_a_app if should_disable_filter else oef_V(E_a_app)  # One Euro filtered voltage
            next_ctrl_t += dt_c

        # Save true states
        theta_arr[i] = x[0]
        omega_arr[i] = omega_true
        i_a_arr[i]   = i_a_true
        e_int_arr[i] = e_int
        # Save noisy measurements and voltages
        omega_meas_arr[i] = omega_meas
        i_a_meas_arr[i]   = i_a_meas
        E_a_desired[i]  = E_a_des
        E_a_applied[i]  = E_a_app
        E_a_filtered[i] = E_a_filt

        if i < N - 1:
            # Integrate error using noisy measurement (forward Euler)
            e_int += (r(t) - i_a_meas) * dt
            x = rk4_step(x, E_a_filt, dt)   # motor sees filtered voltage

    r_vals = r(t_eval)

    di_a_dt = np.diff(i_a_arr) / dt
    print(f"Max rate of change in true current: {np.max(np.abs(di_a_dt)):.2f} A/s")

    t_eval = t_eval * 1000

    TITLE_SIZE  = 30
    LABEL_SIZE  = 22
    TICK_SIZE   = 22
    LEGEND_SIZE = 20
    LINE_WIDTH  = 3
    LINE_NOISE_WIDTH = 0.8

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle('Current control - P + FF', fontsize=TITLE_SIZE)

    axes[0].plot(t_eval, r_vals, 'r--', linewidth=LINE_WIDTH, label='Reference [A]')
    axes[0].plot(t_eval, i_a_meas_arr, color='tab:orange', linewidth=0.8, label=r'$I_a$ measured')
    axes[0].plot(t_eval, i_a_arr, color='tab:blue', linewidth=LINE_WIDTH, label=r'$I_a$ true')
    axes[0].set_ylabel('Current [A]', fontsize=LABEL_SIZE)
    axes[0].tick_params(axis='both', labelsize=TICK_SIZE)
    axes[0].legend(fontsize=LEGEND_SIZE, loc='lower right')
    axes[0].grid()

    # axes[1].plot(t_eval, V_desired, '--', label=r'$V_\mathrm{desired}$ (unclamped)')
    axes[1].axhline( E_a_sat, color='red', linestyle='--', linewidth=LINE_WIDTH, label=r'$E_{a,\mathrm{sat}}=\pm12V$')
    axes[1].axhline(-E_a_sat, color='red', linestyle='--', linewidth=LINE_WIDTH)
    axes[1].plot(t_eval, E_a_applied,  color='tab:orange', linewidth=LINE_WIDTH, label=r'$E_a$')
    if not should_disable_filter:
        axes[1].plot(t_eval, E_a_filtered, color='tab:blue', linewidth=LINE_WIDTH, label=r'$E_a$ filtered (1€)')
    axes[1].set_ylabel(r'$E_a$ [V]', fontsize=LABEL_SIZE)
    axes[1].tick_params(axis='both', labelsize=TICK_SIZE)
    axes[1].legend(fontsize=LEGEND_SIZE, loc='lower right')
    axes[1].grid()

    axes[2].plot(t_eval, omega_meas_arr, color='C0', alpha=0.4, linewidth=LINE_NOISE_WIDTH, label=r'$\omega$ estimated')
    axes[2].plot(t_eval, omega_arr, color='C0', linewidth=LINE_WIDTH, label=r'$\omega$ true (rad/s)')
    axes[2].set_xlabel('Time [ms]', fontsize=LABEL_SIZE)
    axes[2].set_ylabel(r'Ang. vel. [rad/s]', fontsize=LABEL_SIZE)
    axes[2].tick_params(axis='both', labelsize=TICK_SIZE)
    axes[2].legend(fontsize=LEGEND_SIZE, loc='lower right')
    axes[2].grid()

    plt.tight_layout()
    plt.savefig(r'C:\Users\Anders\OneDrive - NTNU\Fordypningsoppgave - Nsquared\finger_dynamics\Fordypningsoppgave_Nsquared\adaptive_control\figures\motor_current_control_feed_forward_PI.pdf', bbox_inches='tight')
    plt.savefig(r'C:\Users\Anders\OneDrive - NTNU\Fordypningsoppgave - Nsquared\specialization_project\Thesis\Figures\Adaptive_Controller\motor_current_control_feed_forward_PI.pdf', bbox_inches='tight')
