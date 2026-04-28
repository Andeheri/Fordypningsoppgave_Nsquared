import numpy as np
from numpy import sin, pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov
from tqdm import tqdm
import os

"""
Define motor parameters
"""
V_supply = 12.0   # V
g = 9.81

# Information from datasheet: https://www.pololu.com/file/0J1829/pololu-25d-metal-gearmotors.pdf, page 3
Ia_stall = 4.9  # A
tau_stall = 220 * g / 1000  # Nm (stall torque at 12 V)

Ia_no_load = 0.2  # A
theta_dot_no_load = 130 * 2 * pi / 60  # rad/s

Kt = tau_stall / Ia_stall
Ra = V_supply / Ia_stall
Bm = Kt * Ia_no_load / theta_dot_no_load
Kb = (V_supply - Ra * Ia_no_load) / theta_dot_no_load

# Motor/electrical parameters
Jm = 0.093   # kg*m^2
La = 0.006   # H

# PWM parameters
pwm_frequency   = 1000.0          # Hz
pwm_period      = 1.0 / pwm_frequency
voltage_turnoff = 1.0             # V – suppress switching at near-zero voltages

Ia_max = Ia_stall  # A (max current magnitude, for clamping)

# Unknown mechanical load parameters (used only in the simulated plant)
J = 0.5
c = 0.2
k = 0.5
theta_eq = pi / 6

"""
Reduced 2-state plant:
x = [theta, omega]^T
u = i_cmd   (assume fast inner current loop, so actual current tracks command)
"""
M = Jm + J

A_true = np.array([
    [0.0, 1.0],
    [-k / M, -(Bm + c) / M]
])

B_true = np.array([
    [0.0],
    [Kt / M]
])

d_true = np.array([
    0.0,
    k * theta_eq / M
])

"""
Internal model
"""
A_ = np.array([
    [0.0, 1.0],
    [0.0, -Bm / Jm]
])

B_ = np.array([
    [0.0],
    [Kt / Jm]
])

"""
Reference model
Choose desired theta-tracking dynamics here
"""
omega_n = 5.0
zeta = 1.0

A_m = np.array([
    [0.0, 1.0],
    [-omega_n**2, -2.0 * zeta * omega_n]
])

B_m = np.array([
    [0.0],
    [omega_n**2]
])

Q = np.diag([10.0, 1.0])
P = solve_continuous_lyapunov(A_m.T, -Q)

print("P matrix:")
print(P)

"""
Ideal matching parameters (only for verification/debugging)
Controller does NOT use these
"""
K_star = np.linalg.pinv(B_true) @ (A_true - A_m)
L_star = float(np.squeeze(np.linalg.pinv(B_true) @ B_m))

K_0 = (np.linalg.pinv(B_true) @ A_m).flatten()
L_0 = float(np.squeeze(np.linalg.pinv(B_m) @ B_m))

K_0 = (np.linalg.pinv(B_) @ (A_ - A_m)).flatten().tolist()
L_0 = float(np.squeeze(np.linalg.pinv(B_) @ B_m))

K_0 = [0.0, 0.0]
L_0 = 1.0


print("K_star =", K_star)
print("L_star =", L_star)

"""
Adaptive gains
"""
Gamma_K = np.diag([20.0, 3.0])   # adaptation rate for K = [K1, K2]
gamma_L = 20.0 * 1.0                  # adaptation rate for L

# PI current controller (inner loop)
# Gains tuned for bandwidth >> mechanical natural frequency (omega_n = 5 rad/s)
Kp_i = 1.0    # proportional gain
Ki_i = 200.0  # integral gain

"""
Reference input
"""
t0 = 1.0
r_max = pi / 4
r = lambda t: r_max * (t > t0)
r = lambda t: r_max * sin(2 * pi * 0.5 * t)
# Square wave reference:
r = lambda t: r_max * (sin(2 * pi * 0.5 * t) > 0).astype(float)


def current_clamp(i_cmd):
    return float(np.clip(i_cmd, -Ia_stall, Ia_stall))


def pwm_voltage(t, V_ideal):
    """Convert an ideal voltage to a PWM signal (scalar inputs).

    Duty cycle d = |V_ideal| / V_supply.
    Positive V_ideal → switches between +V_supply and 0.
    Negative V_ideal → switches between -V_supply and 0.
    Outputs 0 when |V_ideal| < voltage_turnoff.
    """
    if abs(V_ideal) < voltage_turnoff:
        return 0.0
    d = abs(V_ideal) / V_supply
    phase = (t % pwm_period) / pwm_period
    active = float(np.sign(V_ideal)) * V_supply
    return active if phase < d else 0.0


def control_law(x: np.ndarray, r_val: float, K: np.ndarray, L: float):
    i_cmd = -K @ x + L * r_val
    return current_clamp(i_cmd)


def closed_loop_dynamics(t, z):
    """
    State:
    z = [theta, omega, i_a, e_int_curr, theta_m, omega_m, K1, K2, L]

    Architecture:
      - MRAC outer loop  : computes desired current i_cmd = -K@x + L*r
      - PI inner loop    : drives i_a → i_cmd via voltage command
      - PWM              : converts voltage command to switched supply voltage
      - Electrical plant : di_a = (V_pwm - Ra*i_a - Kb*omega) / La
      - Mechanical plant : driven by actual i_a (not i_cmd)
      - Adaptive law     : sees i_cmd as the nominal input (two-timescale separation)
    """
    theta, omega, i_a, e_int_curr = z[0], z[1], z[2], z[3]
    xm = z[4:6]
    K  = z[6:8]
    L  = z[8]

    x   = np.array([theta, omega])
    r_t = r(t)
    e   = x - xm

    # MRAC outer loop: desired current command
    i_cmd = control_law(x, r_t, K, L)

    # Inner PI current controller → voltage command
    curr_err = i_cmd - i_a
    V_ideal  = float(np.clip(Kp_i * curr_err + Ki_i * e_int_curr, -V_supply, V_supply))
    V        = pwm_voltage(t, V_ideal)

    # Mechanical plant (driven by actual current i_a)
    dtheta = omega
    domega = float(A_true[1, 0] * theta + A_true[1, 1] * omega + B_true[1, 0] * i_a + d_true[1])

    # Electrical plant
    di_a        = (V - Ra * i_a - Kb * omega) / La
    de_int_curr = curr_err

    # Reference model
    dxm = A_m @ xm + B_m.flatten() * r_t

    # MRAC adaptive law (uses i_cmd as nominal control input)
    sigma = (B_m.T @ P @ e.reshape(-1, 1)).item()
    dK    = Gamma_K @ x * sigma
    dL    = -gamma_L * r_t * sigma

    return np.hstack(([dtheta, domega, di_a, de_int_curr], dxm, dK, [dL]))


if __name__ == "__main__":
    """
    Simulation parameters
    """
    T = 20.0
    N = 5000
    t_eval = np.linspace(0, T, N)

    # [theta, omega, i_a, e_int_curr, theta_m, omega_m, K1, K2, L]
    z0 = [
        0.0, 0.0, 0.0, 0.0,   # plant: theta, omega, i_a; PI integrator
        0.0, 0.0,              # reference model
        K_0[0], K_0[1],        # adaptive K
        L_0                    # adaptive L
    ]

    save_folder = "adaptive_control/figures"
    filename = "mrac_ideal_mass_spring_damper_test"
    os.makedirs(save_folder, exist_ok=True)

    with tqdm(total=T, desc="Simulating", unit="s", dynamic_ncols=True) as pbar:
        last_t = [0.0]

        def ode_with_progress(t, z):
            pbar.update(t - last_t[0])
            last_t[0] = t
            return closed_loop_dynamics(t, z)

        sol = solve_ivp(
            ode_with_progress,
            t_span=[0.0, T],
            y0=z0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )

    # Extract results
    theta   = sol.y[0]
    omega   = sol.y[1]
    i_a     = sol.y[2]
    # e_int_curr = sol.y[3]   (not plotted)
    theta_m = sol.y[4]
    omega_m = sol.y[5]
    K1      = sol.y[6]
    K2      = sol.y[7]
    L       = sol.y[8]

    print("Adaptive gains at final time:")
    print(f"K_0 = [{float(K1[-1])}, {float(K2[-1])}]")
    print(f"L_0 = {L[-1]}")

    x_all  = sol.y[0:2]
    r_all  = r(t_eval)
    i_cmd  = np.clip(-(K1 * x_all[0] + K2 * x_all[1]) + L * r_all, -Ia_stall, Ia_stall)

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)

    def setup_axis(ax, ylabel, title=None):
        if title:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    # Theta tracking
    axes[0].plot(t_eval, theta_m, '--', label=r'$\theta_m$')
    axes[0].plot(t_eval, theta, label=r'$\theta$')
    axes[0].plot(t_eval, r_all, 'r--', alpha=0.5, label='r')
    axes[0].legend()
    setup_axis(axes[0], 'Angle (rad)', title='Angle Tracking')

    # Current tracking (inner loop)
    axes[1].plot(t_eval, i_cmd, '--', label=r'$i_{cmd}$ (MRAC)')
    axes[1].plot(t_eval, i_a,         label=r'$i_a$ (actual)')
    axes[1].axhline( Ia_stall, color='gray', linestyle=':', linewidth=0.8)
    axes[1].axhline(-Ia_stall, color='gray', linestyle=':', linewidth=0.8)
    axes[1].legend()
    setup_axis(axes[1], 'Current (A)', title='Current Tracking (PI inner loop)')

    # Adaptive gains
    axes[2].plot(t_eval, K1, label='K1')
    axes[2].plot(t_eval, K2, label='K2')
    axes[2].plot(t_eval, L,  label='L')
    # axes[2].axhline(K_star[0, 0], color='C0', linestyle='--', alpha=0.7, label='K1*')
    # axes[2].axhline(K_star[0, 1], color='C1', linestyle='--', alpha=0.7, label='K2*')
    # axes[2].axhline(L_star,       color='C2', linestyle='--', alpha=0.7, label='L*')
    axes[2].legend()
    setup_axis(axes[2], 'Gain', title='Adaptive Gains')
    axes[2].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(f"{save_folder}/{filename}.png", dpi=300)

    plt.show()