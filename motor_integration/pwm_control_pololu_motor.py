
import numpy as np
from numpy import sin, cos, pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

"""
Simulation parameters
"""
T = 5.0  # Total simulation time (s)
N = 1000  # Number of time steps
t_eval = np.linspace(0, T, N)
dt = t_eval[1] - t_eval[0]
save_folder_path = "motor_integration/figures"
filename = "pi_pd_pwm_pololu_control_results.png"

theta_0 = 0.0  # Initial angle (rad)
omega_0 = 0.0  # Initial angular velocity (rad/s)
i_a_0 = 0.0    # Initial armature current (A)

"""
Define motor parameters
"""
V_supply = 12.0   # V (Supply voltage / saturation limit)
pwm_frequency = 1000.0  # Hz (PWM frequency)
pwm_period = 1.0 / pwm_frequency  # s (PWM period)
voltage_turnoff = 1.0  # Absolute voltage level where the PWM output should switch to zero to avoid excessive switching at low voltages

g = 9.81  # m/s^2 (Gravitational acceleration)
# Information from datasheet: https://www.pololu.com/file/0J1829/pololu-25d-metal-gearmotors.pdf, page 3
Ia_stall = 4.9  # A
tau_stall = 220 * g / 1000 # Nm (Stall torque at 12 V)

Ia_no_load = 0.2  # A (No-load current at 12 V)
theta_dot_no_load = 130 * 2 * pi / 60  # rad/s (No-load speed at 12 V)

Kt = tau_stall / Ia_stall # Nm/A (Torque constant) ≈ 0.44 Nm/A
Ra = V_supply / Ia_stall # Ω (Armature resistance) ≈ 2.44 Ω (Measured from stall current at 12 V)
Bm = Kt * Ia_no_load / theta_dot_no_load  # Nm*s (Rotor friction coefficient)
Kb = (V_supply - Ra * Ia_no_load) / theta_dot_no_load   # V*s/rad (Back EMF constant)

# Unkowns
Jm = 0.093  # kg*m^2 (Rotor moment of inertia)
La = 0.006  # H (Armature inductance)

print("Motor parameters:")
print(f"  Jm = {Jm} kg*m^2")
print(f"  Bm = {Bm} Nm*s")
print(f"  Kb = {Kb} V*s/rad")
print(f"  Kt = {Kt} Nm/A")
print(f"  Ra = {Ra:.2f} Ω")
print(f"  La = {La} H")
print(Kt * 0.87 / g * 1000)

"""
PI-PD control parameters

Control law (2-DOF structure):
  V = Kp1*(r - θ) + Ki*∫(r - θ)dt  [PI on error]
    - Kp2*ω - Kd*(Kt*i_a - Bm*ω)/Jm [PD feedback on output]
"""
Kp1 = 50.0
Ki  = 3.0

Kp2 = 2.0
Kd  = 5

t0 = 1.0
r_max = 5.0
r = lambda t: r_max * sin(2 * pi * 0.5 * t)  # Sinusoidal reference input (0.5 Hz)

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
    return np.clip(V, -V_supply, V_supply)


def control_law(theta, theta_error, omega, e_int):
    V = Kp1*theta_error + Ki*e_int - (Kp2 * theta + Kd * omega)
    return voltage_clamp(V)


def pwm_voltage(t, V_ideal):
    """Convert an ideal voltage to a unipolar PWM signal.

    Duty cycle: d = |V_ideal| / V_supply  ∈ [0, 1]
    Active level: sign(V_ideal) * V_supply
    Off level: 0 V

    Positive V_ideal → switches between +V_supply and 0.
    Negative V_ideal → switches between -V_supply and 0.
    Works with both scalars and numpy arrays.
    """
    d = np.abs(V_ideal) / V_supply                 # duty cycle ∈ [0, 1]
    phase = (t % pwm_period) / pwm_period       # phase within current period ∈ [0, 1)
    active = np.sign(V_ideal) * V_supply
    pwm = np.where(phase < d, active, 0.0)
    return np.where(np.abs(V_ideal) < voltage_turnoff, 0.0, pwm)  # zero output at low voltages


def closed_loop_dynamics(t, x):
    theta, omega, i_a, e_int = x
    theta_error = r(t) - theta
    V_ideal = control_law(theta, theta_error, omega, e_int)
    V = pwm_voltage(t, V_ideal)
    dtheta  = omega
    domega  = (Kt * i_a - Bm * omega) / Jm
    di_a    = (V - Ra * i_a - Kb * omega) / La
    de_int  = theta_error          # d/dt ∫e dt = e
    return [dtheta, domega, di_a, de_int]



if __name__ == "__main__":
    # Simulate the closed-loop system
    sol = solve_ivp(closed_loop_dynamics, t_span=[0, T], y0=[theta_0, omega_0, i_a_0, 0.0], t_eval=t_eval)

    # Extract results
    theta = sol.y[0]
    omega = sol.y[1]
    i_a   = sol.y[2]
    e_int = sol.y[3]

    # Compute control input over time for plotting
    V_ideal = control_law(theta, r(t_eval) - theta, omega, e_int)

    # Dense time grid for PWM plot: 20 samples per PWM period to show switching clearly
    t_pwm = np.linspace(0, T, 20 * int(T * pwm_frequency))
    V_ideal_dense = np.interp(t_pwm, t_eval, V_ideal)
    pwm_signal = pwm_voltage(t_pwm, V_ideal_dense)

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1, 1]})

    def setup_axis(ax, ylabel, color='black', title=None):
        if title:
            ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.grid()

    axes[0].plot(t_eval, theta, label='θ (rad)')
    axes[0].plot(t_eval, r(t_eval), 'r--', label='Reference (rad)')
    axes[0].legend()
    setup_axis(axes[0], 'Angle (rad)', title='Joint Angle θ')

    axes[1].plot(t_eval, i_a, color='tab:blue', label='i_a (A)')
    axes[1].legend(loc='upper left')
    setup_axis(axes[1], 'Current (A)', color='tab:blue', title='Armature Current')

    # Compute PWM average: box-filter over one PWM period
    samples_per_period = 20  # set above when building t_pwm
    kernel = np.ones(samples_per_period) / samples_per_period
    pwm_average = np.convolve(pwm_signal, kernel, mode='same')

    axes[2].plot(t_pwm, pwm_signal, color='tab:orange', label='V PWM (V)', linewidth=0.5, alpha=0.5)
    axes[2].plot(t_eval, V_ideal, color='tab:green', label='V ideal (V)', linewidth=1.5)
    axes[2].plot(t_pwm, pwm_average, color='tab:red', label='V avg (V)', linewidth=1.5, linestyle='--')
    axes[2].axhline(V_supply, color='gray', linestyle=':', label=f'V_supply = ±{V_supply} V')
    axes[2].axhline(-V_supply, color='gray', linestyle=':')
    axes[2].axhline(voltage_turnoff, color='purple', linestyle='--', linewidth=1.0, label=f'V_turnoff = ±{voltage_turnoff} V')
    axes[2].axhline(-voltage_turnoff, color='purple', linestyle='--', linewidth=1.0)
    axes[2].legend(loc='upper left')
    setup_axis(axes[2], 'Voltage (V)', color='tab:orange', title='PWM Control Signal')

    plt.tight_layout()
    plt.savefig(f"{save_folder_path}/{filename}")

    # Save the PWM subplot separately
    fig_pwm, ax_pwm = plt.subplots(figsize=(12, 4))
    ax_pwm.plot(t_pwm, pwm_signal, color='tab:orange', label='V PWM (V)', linewidth=0.5, alpha=0.5)
    ax_pwm.plot(t_eval, V_ideal, color='tab:green', label='V ideal (V)', linewidth=1.5)
    ax_pwm.plot(t_pwm, pwm_average, color='tab:red', label='V avg (V)', linewidth=1.5, linestyle='--')
    ax_pwm.axhline(V_supply, color='gray', linestyle=':', label=f'V_supply = ±{V_supply} V')
    ax_pwm.axhline(-V_supply, color='gray', linestyle=':')
    ax_pwm.axhline(voltage_turnoff, color='purple', linestyle='--', linewidth=1.0, label=f'V_turnoff = ±{voltage_turnoff} V')
    ax_pwm.axhline(-voltage_turnoff, color='purple', linestyle='--', linewidth=1.0)
    ax_pwm.legend(loc='upper left')
    setup_axis(ax_pwm, 'Voltage (V)', title='PWM Control Signal')
    fig_pwm.tight_layout()
    fig_pwm.savefig(f"{save_folder_path}/pwm_signal.png", dpi = 600)
    


