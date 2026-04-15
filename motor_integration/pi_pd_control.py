
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

"""
Simulation parameters
"""
T = 5.0  # Total simulation time (s)
N = 1000  # Number of time steps
t_eval = np.linspace(0, T, N)
dt = t_eval[1] - t_eval[0]

theta_0 = 0.0  # Initial angle (rad)
omega_0 = 0.0  # Initial angular velocity (rad/s)
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
Kp1 = 6.0
Ki  = 5.0

Kp2 = 1.0
Kd  = 0.65

t0 = 1.0
r_max = 5.0

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


def control_law(theta, theta_error, omega, e_int):
    V = Kp1*theta_error + Ki*e_int - (Kp2 * theta + Kd * omega)
    return voltage_clamp(V)


def closed_loop_dynamics(t, x):
    theta, omega, i_a, e_int = x
    theta_error = r(t) - theta
    V = control_law(theta, theta_error, omega, e_int)
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
    control_input = [control_law(theta[i], r(t_eval[i]) - theta[i], omega[i], e_int[i]) for i in range(len(t_eval))]

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(t_eval, theta, label='θ (rad)')
    axes[0].plot(t_eval, r(t_eval), 'r--', label='Reference (rad)')
    axes[0].set_title('Joint Angle θ')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Angle (rad)')
    axes[0].legend()
    axes[0].grid()

    ax_i = axes[1]
    ax_v = ax_i.twinx()

    l1, = ax_i.plot(t_eval, i_a, color='tab:blue', label='i_a (A)')
    l2, = ax_v.plot(t_eval, control_input, color='tab:orange', label='V (V)')
    l3 = ax_v.axhline(V_sat, color='tab:red', linestyle='--', label=f'V_sat = ±{V_sat} V')
    ax_v.axhline(-V_sat, color='tab:red', linestyle='--')

    ax_i.set_title('Armature Current & Control Input Voltage')
    ax_i.set_xlabel('Time (s)')
    ax_i.set_ylabel('Current (A)', color='tab:blue')
    ax_v.set_ylabel('Voltage (V)', color='tab:orange')
    ax_i.tick_params(axis='y', labelcolor='tab:blue')
    ax_v.tick_params(axis='y', labelcolor='tab:orange')
    ax_i.legend(handles=[l1, l2, l3], loc='upper left')
    ax_i.grid()

    plt.tight_layout()
    plt.show()


