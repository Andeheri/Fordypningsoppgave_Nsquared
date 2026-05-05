
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Simulation parameters
"""
T  = 8*10**-3       # Total simulation time (s)
fs = 100_000   # Sample rate (Hz)  – fixed step, fully transparent
dt = 1.0 / fs
N  = int(T * fs) + 1
t_eval = np.linspace(0, T, N)

theta_0 = 0.0  # Initial angle (rad)
omega_0 = 4.0  # Initial angular velocity (rad/s)
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

V_sat = 12.0   # V (Supply voltage / saturation limit)
pwm_frequency = 1000.0  # Hz (PWM frequency)
pwm_period = 1.0 / pwm_frequency  # s (PWM period)

"""
PI-PD control parameters

Control law (2-DOF structure):
  V = Kp1*(r - θ) + Ki*∫(r - θ)dt  [PI on error]
    - Kp2*ω - Kd*(Kt*i_a - Bm*ω)/Jm [PD feedback on output]
"""

Kp = 20
Ki = 800

should_disable_feed_forward = False

t0 = 2*10**-3
r_max = 2.0

# Measurement noise standard deviations (set to 0 to disable)
noise_std_Ia    = 0.05   # A
noise_std_omega = 0.01    # rad/s

r = lambda t: r_max * (t > t0)  # Step reference input (1 rad)


"""
System dynamics

Transfer function:  θ/V = Kt / (Jm*La*s³ + (Jm*Ra + Bm*La)*s² + (Ra*Bm + Kt*Kb)*s)
State-space realisation  x = [θ, ω, i_a]:
  dθ/dt   = ω
  dω/dt   = (Kt*i_a - Bm*ω) / Jm
  di_a/dt = (V - Ra*i_a - Kb*ω) / La
"""


def voltage_clamp(V):
    return np.clip(V, -V_sat, V_sat)


def control_law(i_a, omega, e_int, t):
    r_t = r(t)
    V = Kp * (r_t - i_a) + Ki * e_int
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
    e_int_arr      = np.zeros(N)
    # Noisy measurements seen by the controller
    omega_meas_arr = np.zeros(N)
    i_a_meas_arr   = np.zeros(N)
    V_desired = np.zeros(N)   # unclamped controller output
    V_applied = np.zeros(N)   # what the motor actually sees

    x     = np.array([theta_0, omega_0, i_a_0])
    e_int = 0.0
    rng   = np.random.default_rng(seed=0)

    for i in tqdm(range(N)):
        t = t_eval[i]
        _, omega_true, i_a_true = x

        # Noisy measurements - controller only sees these
        i_a_meas   = i_a_true   + rng.normal(0.0, noise_std_Ia)
        omega_meas = omega_true + rng.normal(0.0, noise_std_omega)

        # Controller evaluated at noisy measurements
        V_des = Kp * (r(t) - i_a_meas) + Ki * e_int + (Kb * omega_meas if not should_disable_feed_forward else 0.0)
        V_app = voltage_clamp(V_des)                         # clamped

        # Save true states
        theta_arr[i] = x[0]
        omega_arr[i] = omega_true
        i_a_arr[i]   = i_a_true
        e_int_arr[i] = e_int
        # Save noisy measurements
        omega_meas_arr[i] = omega_meas
        i_a_meas_arr[i]   = i_a_meas
        V_desired[i] = V_des
        V_applied[i] = V_app

        if i < N - 1:
            # Integrate error using noisy measurement (forward Euler)
            e_int += (r(t) - i_a_meas) * dt
            x = rk4_step(x, V_app, dt)

    r_vals = r(t_eval)
    t_eval = t_eval * 1000

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t_eval, i_a_meas_arr, color='tab:orange', linewidth=0.8, label=r'$I_a$ measured')
    axes[0].plot(t_eval, i_a_arr, color='tab:blue', label=r'$I_a$ true')
    axes[0].plot(t_eval, r_vals, 'r--', label='Reference (A)')
    axes[0].set_title('Armature Current $I_a$')
    axes[0].set_ylabel('Current (A)')
    axes[0].legend()
    axes[0].grid()

    # axes[1].plot(t_eval, V_desired, '--', label=r'$V_\mathrm{desired}$ (unclamped)')
    axes[1].axhline( V_sat, color='red', linestyle='--')
    axes[1].axhline(-V_sat, color='red', linestyle='--')
    axes[1].plot(t_eval, V_applied, label=r'$V_\mathrm{applied}$')
    axes[1].set_title(r'Control Voltage $V$')
    axes[1].set_ylabel('Voltage (V)')
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(t_eval, omega_meas_arr, color='C0', alpha=0.4, linewidth=0.8, label=r'$\omega$ measured')
    axes[2].plot(t_eval, omega_arr, color='C0', label=r'$\omega$ true (rad/s)')
    axes[2].set_title(r'Angular Velocity $\omega$')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Angular Velocity (rad/s)')
    axes[2].legend()
    axes[2].grid()

    plt.tight_layout()
    plt.savefig('motor_current_control_feed_forward_PI.png')


