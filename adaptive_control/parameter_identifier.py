
"""
Adaptive parameter estimation for brushed DC motor connected to unkown mass-spirng-damper load.

Dynamics:
    (Jm + m) q'' + (Bm + c) q' + k (q - q0) = Kt Ia
    Ea = Ra Ia + La Ia' + Kb q'
"""
import numpy as np
from numpy import sin, cos, pi
from scipy.signal import cont2discrete, lfilter, lfilter_zi
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch

"""
-------------------------------- Motor parameters --------------------------------
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
Jm = 0.093  # kg*m^2 (Rotor moment of inertia)
La = 0.006  # H (Armature inductance)  (Not sure if it's correct)

# Load to be estimated:
m = 0.5
c = 0.2
k = 20
q0 = 1.0

"""
------------------------------ Simulation parameters ------------------------------
"""

save_path = "adaptive_control/figures/parameter_estimation_results.png"
dpi = 600

dt = 0.01
T = 5.0
num_steps = int(T / dt)
time = np.linspace(0, T, num_steps)

# Adaptive controller (gain)
gamma = 1.0  # <- small adaptation gain for stability/conditioning

# Parameter covariance (SPD)
P_0 = np.diag(np.array([50.0, 40.0, 20.0, 5.0])).astype(float)

# Initial guesses
m_0 = 0.2
c_0 = 0.1
k_0 = 10.0
q0_0 = 0.5

beta_forget_factor = 2.0
sigma_leak = 1e-4

"""
-------------------------------- Filter parameters --------------------------------
"""
lambda2, lambda1, lambda0 = 20.0, 10.0, 5.0
denominator = [1.0, lambda2, lambda1, lambda0]  # Λ(s) = s^3 + λ2 s^2 + λ1 s + λ0

# z = Kt*Kb*s*theta+Kt*Ea / Λ(s), phi = ]
numeratorz_q = [Kt * Kb, 0.0]  # Kt * Kb * s
numeratorz_Ea = [Kt]               # Kt
numerator1 = [La, Ra, 0.0, 0.0]    # La*s**3 + Ra*s**2
numerator2 = [La, Ra, 0.0]         # La*s**2 + Ra*s
numerator3 = [La, Ra]              # La*s + Ra
numerator4 = [-La, -Ra]            # -La*s - Ra
denominator4 = [1.0, lambda2, lambda1, lambda0, 0.0]  # Λ(s) = s^4 + λ2 s^3 + λ1 s^2 + λ0 s
filter_method = "bilinear"
print("Discretizing filters...")
bz_q, az_q, _ = cont2discrete((numeratorz_q, denominator), dt=dt, method=filter_method)
bz_Ea, az_Ea, _ = cont2discrete((numeratorz_Ea, denominator), dt=dt, method=filter_method)
b1, a1, _ = cont2discrete((numerator1, denominator), dt=dt, method=filter_method)
b2, a2, _ = cont2discrete((numerator2, denominator), dt=dt, method=filter_method)
b3, a3, _ = cont2discrete((numerator3, denominator), dt=dt, method=filter_method)
b4, a4, _ = cont2discrete((numerator4, denominator4), dt=dt, method=filter_method)

bz_q, az_q = bz_q.ravel(), az_q.ravel()
bz_Ea, az_Ea = bz_Ea.ravel(), az_Ea.ravel()
b1, a1 = b1.ravel(), a1.ravel()
b2, a2 = b2.ravel(), a2.ravel()
b3, a3 = b3.ravel(), a3.ravel()
b4, a4 = b4.ravel(), a4.ravel()
zi_z_q    = lfilter_zi(bz_q, az_q) * 0.0
zi_z_Ea   = lfilter_zi(bz_Ea, az_Ea) * 0.0
zi1       = lfilter_zi(b1, a1) * 0.0
zi2       = lfilter_zi(b2, a2) * 0.0
zi3       = lfilter_zi(b3, a3) * 0.0
zi4       = lfilter_zi(b4, a4) * 0.0

"""
---------------------------- Helper functions ----------------------------
"""
def phi(q: float) -> np.ndarray:
    global zi1, zi2, zi3, zi4
    phi1, zi1 = lfilter(b1, a1, [q], zi=zi1)  # ((La*s^3+Ra*s^2)/Λ) [q]
    phi2, zi2 = lfilter(b2, a2, [q], zi=zi2)  # ((La*s^2+Ra*s)/Λ) [q]
    phi3, zi3 = lfilter(b3, a3, [q], zi=zi3)  # ((La*s+Ra)/Λ) [q]
    phi4, zi4 = lfilter(b4, a4, [1.0], zi=zi4)    # ((-La*s-Ra)/(s*Λ)) [1] 
    return np.array([phi1, phi2, phi3, phi4], dtype=float).reshape(-1)

def epsilon(z: float, theta: np.ndarray, phi_v: np.ndarray, ms2: float) -> float:
    return float((z - theta @ phi_v) / ms2)

def ms2_func(phi_v: np.ndarray) -> float:
    return 1.0 + float(phi_v @ phi_v)

def theta_dot(theta: np.ndarray, P: np.ndarray, epsk: float, phi_v: np.ndarray) -> np.ndarray:
    y = gamma * epsk * (P @ phi_v) - sigma_leak * theta
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    return y

def P_dot(P: np.ndarray, phi_v: np.ndarray, ms2: float, beta_forget: float) -> np.ndarray:
    Pphi = P @ phi_v
    return beta_forget * P - np.outer(Pphi, Pphi) / (ms2 + 1e-12)

# Plant dynamics (RK4)
def q_ddot(q_, qd_, Ia_):
    return (-(Bm + c) * qd_ - k * (q_ - q0) + Kt * Ia_) / (Jm + m)

def Ia_dot(q_, qd_, Ia_, Ea_):
    return (Ea_ - Ra * Ia_ - Kb * qd_) / La

def rk4_step(q_, qd_, Ia_, Ea_, h):
    # k1
    k1_q  = qd_
    k1_qd = q_ddot(q_, qd_, Ia_)
    k1_Ia = Ia_dot(q_, qd_, Ia_, Ea_)

    # k2
    q_k2  = q_  + 0.5 * h * k1_q
    qd_k2 = qd_ + 0.5 * h * k1_qd
    Ia_k2 = Ia_ + 0.5 * h * k1_Ia

    k2_q  = qd_k2
    k2_qd = q_ddot(q_k2, qd_k2, Ia_k2)
    k2_Ia = Ia_dot(q_k2, qd_k2, Ia_k2, Ea_)

    # k3
    q_k3  = q_  + 0.5 * h * k2_q
    qd_k3 = qd_ + 0.5 * h * k2_qd
    Ia_k3 = Ia_ + 0.5 * h * k2_Ia

    k3_q  = qd_k3
    k3_qd = q_ddot(q_k3, qd_k3, Ia_k3)
    k3_Ia = Ia_dot(q_k3, qd_k3, Ia_k3, Ea_)

    # k4
    q_k4  = q_  + h * k3_q
    qd_k4 = qd_ + h * k3_qd
    Ia_k4 = Ia_ + h * k3_Ia

    k4_q  = qd_k4
    k4_qd = q_ddot(q_k4, qd_k4, Ia_k4)
    k4_Ia = Ia_dot(q_k4, qd_k4, Ia_k4, Ea_)

    # combine
    q_next  = q_  + (h / 6.0) * (k1_q  + 2*k2_q  + 2*k3_q  + k4_q)
    qd_next = qd_ + (h / 6.0) * (k1_qd + 2*k2_qd + 2*k3_qd + k4_qd)
    Ia_next = Ia_ + (h / 6.0) * (k1_Ia + 2*k2_Ia + 2*k3_Ia + k4_Ia)

    return q_next, qd_next, Ia_next

# ---------------------- Arrays & initial conditions ----------------------
q      = np.zeros(num_steps)
qd     = np.zeros(num_steps)
Ia     = np.zeros(num_steps)


# Persistently exciting input
Ea = V_supply * np.sign(sin(2 * pi * 0.5 * time))  # Square wave at 0.5 Hz, amplitude equal to supply voltage

z_arr  = np.zeros(num_steps)
eps    = np.zeros(num_steps)
ms2    = np.zeros(num_steps)

phi_log   = np.zeros((4, num_steps))
theta     = np.zeros((4, num_steps))

theta[:, 0] = np.array([Jm + m_0, Bm + c_0, k_0, k_0 * q0_0], dtype=float)
P = P_0.copy()
print("Initial parameter estimates:")
# ------------------------------- Simulation -------------------------------
for k in tqdm(range(num_steps-1), desc="Simulating"):
    # plant
    q[k+1], qd[k+1], Ia[k+1] = rk4_step(q[k], qd[k], Ia[k], Ea[k], dt)

    # filters
    z_k_q_arr, zi_z_q = lfilter(bz_q, az_q, [q[k+1]], zi=zi_z_q)
    z_arr[k+1] = z_k_q_arr[0]
    z_k_Ea_arr, zi_z_Ea = lfilter(bz_Ea, az_Ea, [Ea[k+1]], zi=zi_z_Ea)
    z_arr[k+1] += z_k_Ea_arr[0]

    # use same-time signals for φ
    phi_k = phi(q[k+1])
    phi_log[:, k+1] = phi_k

    ms2[k+1] = ms2_func(phi_k)
    eps[k+1] = epsilon(z_arr[k+1], theta[:, k], phi_k, ms2[k+1])

    th_dot = theta_dot(theta[:, k], P, eps[k+1], phi_k)
    Pd     = P_dot(P, phi_k, ms2[k+1], beta_forget_factor)

    theta[:, k+1] = theta[:, k] + dt * th_dot

    if abs(eps[k+1]) < 1e-6:
        eps[k+1] = 0.0

    P = 0.5 * (P + P.T + dt * (Pd + Pd.T))  # keep symmetric

# ----------------------------- Plot results -----------------------------
eps_den = 1e-9
m_hat    = theta[0, :] - Jm
c_hat = theta[1, :] - Bm
k_hat    = theta[2, :]
q0_hat   = theta[3, :] / k

print(f"Mass: {m_hat[-1]}")
print(f"Damping: {c_hat[-1]}")
print(f"Stiffness: {k_hat[-1]}")
print(f"Equilibrium position: {q0_hat[-1]}")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))

# Mass subplot
ax1.plot(time, m_hat, 'b-', label=r"$\hat m$")
ax1.plot(time, np.full_like(time, m), 'k--', label=r"$m^*$")
ax1.set_ylabel("Mass [kg]")
ax1.grid(True)
ax1.legend()

# Dampening subplot  
ax2.plot(time, c_hat, 'r-', label=r"$\hat c$")
ax2.plot(time, np.full_like(time, c), 'k--', label=r"$c^*$")
ax2.set_ylabel("Damping [Ns/m]")
ax2.grid(True)
ax2.legend()

# Spring subplot
ax3.plot(time, k_hat, 'g-', label=r"$\hat k$")
ax3.plot(time, np.full_like(time, k), 'k--', label=r"$k^*$")
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Stiffness [N/m]")
ax3.grid(True)
ax3.legend()

# Equilibrium position subplot
ax4.plot(time, q0_hat, 'm-', label=r"$\hat q_0$")
ax4.plot(time, np.full_like(time, q0), 'k--', label=r"$q_0^*$")
ax4.set_xlabel("Time [s]")
ax4.set_ylabel("Equilibrium position [m]")
ax4.grid(True)
ax4.legend()

plt.tight_layout()
plt.savefig(save_path, dpi=dpi)
print(f"Saved PNG to: {save_path}")
plt.show()
