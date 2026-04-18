
import numpy as np
from numpy import pi
from scipy.integrate import solve_ivp
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

Jm = 0.093   # kg·m²  (rotor inertia)
La = 0.006   # H      (true armature inductance – unknown to estimator)

# ─── PI-PD controller (identical to pwm_control_pololu_motor.py) ─────────────
Kp1, Ki, Kp2, Kd = 50.0, 3.0, 2.0, 5.0
t0, r_max = 1.0, 5.0
r = lambda t: r_max * (t > t0)   # step reference [rad]

# ─── Adaptive estimator ───────────────────────────────────────────────────────
# Electrical equation filtered by 1/γ, γ = s + λ₀ :
#
#   z  = 1/γ [V]  −  1/γ [Ra·Ia]  −  s/γ [Kb·θ_angle]      (measured)
#   Φ  = s/γ [Ia]                                             (regressor)
#   θ* = La                                                   (unknown)
#
#   SPM:  z = La · Φ
#
# Gradient estimator (no normalisation):
#   dLa_est/dt = Γ · ε · Φ,   ε = z − La_est · Φ
#
# Filter realisations (1/(s+λ₀)[x]  →  ẋ_f = −λ₀·x_f + x):
#   f1  →  1/(s+λ₀)[V]
#   f2  →  1/(s+λ₀)[Ra·Ia]
#   f3  →  1/(s+λ₀)[Kb·θ_angle]   ⟹  s/(s+λ₀)[Kb·θ_angle] = Kb·θ_angle − λ₀·f3
#   f4  →  1/(s+λ₀)[Ia]           ⟹  Φ = Ia − λ₀·f4

lambda0  = 10.0   # filter pole λ₀ > 0
Gamma    = 10.0   # adaptation gain Γ > 0
La_est_0 = 0.001  # initial estimate [H]

# ─── Simulation parameters ────────────────────────────────────────────────────
T      = 2.0
t_eval = np.linspace(0, T, 5_000)

save_folder = "adaptive_control/figures"
filename    = "inductance_estimator_gradient_results.png"


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
# State vector x = [theta_angle, omega, i_a, e_int,  f1, f2, f3, f4,  La_est]
#                    0            1      2    3        4   5   6   7    8

def dynamics(t, x):
    theta_angle, omega, i_a, e_int = x[0], x[1], x[2], x[3]
    f1, f2, f3, f4                 = x[4], x[5], x[6], x[7]
    La_est                         = x[8]

    # Control
    theta_err = r(t) - theta_angle
    V_ideal   = control_law(theta_angle, theta_err, omega, e_int)
    V         = pwm_voltage(t, V_ideal)

    # Motor dynamics
    dtheta = omega
    domega = (Kt * i_a - Bm * omega) / Jm
    di_a   = (V - Ra * i_a - Kb * omega) / La   # true La used here
    de_int = theta_err

    # Filter ODEs
    df1 = -lambda0 * f1 + V
    df2 = -lambda0 * f2 + Ra * i_a
    df3 = -lambda0 * f3 + Kb * theta_angle
    df4 = -lambda0 * f4 + i_a

    # SPM signals
    Phi     = i_a           - lambda0 * f4    # s/(s+λ₀)[Ia]
    z       = f1 - f2 - (Kb * theta_angle - lambda0 * f3)  # z = La·Φ

    # Estimator
    epsilon = z - La_est * Phi
    dLa_est = Gamma * epsilon * Phi

    return [dtheta, domega, di_a, de_int, df1, df2, df3, df4, dLa_est]


if __name__ == "__main__":
    x0 = [0.0, 0.0, 0.0, 0.0,   # theta_angle, omega, i_a, e_int
          0.0, 0.0, 0.0, 0.0,   # f1, f2, f3, f4
          La_est_0]              # La_est

    with tqdm(total=T, desc='Simulating', unit='s', bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} s [{elapsed}<{remaining}]') as pbar:
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
    La_est_sol = sol.y[8]

    # Compute voltage signals for plotting
    V_ideal = control_law(theta_sol, r(t_eval) - theta_sol, omega_sol, e_int_sol)

    # Dense time grid for PWM: 20 samples per PWM period
    t_pwm         = np.linspace(0, T, 20 * int(T * pwm_frequency))
    V_ideal_dense = np.interp(t_pwm, t_eval, V_ideal)
    pwm_signal    = pwm_voltage(t_pwm, V_ideal_dense)

    samples_per_period = 20
    kernel      = np.ones(samples_per_period) / samples_per_period
    pwm_average = np.convolve(pwm_signal, kernel, mode='same')

    # ─── Plots ────────────────────────────────────────────────────────────────
    def setup_axis(ax, ylabel, color='black', title=None):
        if title:
            ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.grid()

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2, 2, 1.5, 1.5]})

    axes[0].plot(t_eval, La_est_sol * 1e3, label=r'$\hat{L}_a$ (mH)')
    axes[0].axhline(La * 1e3, color='r', linestyle='--',
                    label=rf'$L_a$ = {La * 1e3:.1f} mH  (true)')
    axes[0].legend()
    setup_axis(axes[0], 'Inductance (mH)', title='Armature Inductance Estimate')

    axes[1].plot(t_eval, theta_sol,        label='θ (rad)')
    axes[1].plot(t_eval, r(t_eval), 'r--', label='Reference (rad)')
    axes[1].legend()
    setup_axis(axes[1], 'Angle (rad)', title='Joint Angle vs Reference')

    axes[2].plot(t_eval, i_a_sol, color='tab:blue', label='$i_a$ (A)')
    axes[2].legend(loc='upper left')
    setup_axis(axes[2], 'Current (A)', color='tab:blue', title='Armature Current')

    axes[3].plot(t_pwm, pwm_signal,  color='tab:orange', label='V PWM (V)',  linewidth=0.5, alpha=0.5)
    axes[3].plot(t_eval, V_ideal,    color='tab:green',  label='V ideal (V)', linewidth=1.5)
    axes[3].plot(t_pwm, pwm_average, color='tab:red',    label='V avg (V)',   linewidth=1.5, linestyle='--')
    axes[3].axhline( V_supply,        color='gray',   linestyle=':',  label=f'V_supply = ±{V_supply} V')
    axes[3].axhline(-V_supply,        color='gray',   linestyle=':')
    axes[3].axhline( voltage_turnoff, color='purple', linestyle='--', linewidth=1.0, label=f'V_turnoff = ±{voltage_turnoff} V')
    axes[3].axhline(-voltage_turnoff, color='purple', linestyle='--', linewidth=1.0)
    axes[3].legend(loc='upper left')
    setup_axis(axes[3], 'Voltage (V)', color='tab:orange', title='PWM Control Signal')

    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(f"{save_folder}/{filename}", dpi=150)
    plt.show()