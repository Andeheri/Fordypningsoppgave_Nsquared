import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
from tqdm import tqdm
from types import SimpleNamespace
from visualization_finger_simulation_v3 import animate_finger_simulation, plot_simulation_angles, _move_to_secondary
from dynamics import *

"""
----------------------- Motor dynamics -----------------------
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

r_spindle = 0.01  # m (effective radius of the spindle that the cable winds around)
should_use_pure_pd_control = False  # Set to True to disable MRAC adaptation and use only a fixed PD controller (for testing)

"""
Reduced 2-state plant:
x = [theta, omega]^T
u = i_cmd   (assume fast inner current loop, so actual current tracks command)
"""

A_true = np.array([
    [0.0, 1.0],
    [0.0, -Bm / Jm]
])

B_true = np.array([
    [0.0],
    [Kt / Jm]
])

d_true = lambda tau: np.array([
    0.0,
    tau / Jm
])

"""
Reference model
Choose desired theta-tracking dynamics here
"""
omega_n = 8.0
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

K_0 = [0.0, 0.0]
L_0 = 1.0


print("K_star =", K_star)
print("L_star =", L_star)

"""
No load controller
"""
Kp = 13.50
Kd = 3.12

"""
Adaptive gains
"""
should_disable_leakage = True
Gamma_K_f = np.diag([20.0, 3.0]) *0.1  # adaptation rate for K = [K1, K2]
gamma_L_f = 20.0 * 0.1                  # adaptation rate for L
sigma_mod_f = 0.01                      # sigma-modification leakage (prevents parameter drift)

Gamma_K_e = np.diag([30.0, 5.0])*0.1   # adaptation rate for K = [K1, K2]
gamma_L_e = 20.0 *0.1                  # adaptation rate for L
sigma_mod_e = 0.01                      # sigma-modification leakage (prevents parameter drift)

if should_disable_leakage:
    sigma_mod_f = 0.0
    sigma_mod_e = 0.0

"""
Reference input
"""
from enum import Enum

class MovementState(Enum):
    FLEXION   = "FLEXION"
    EXTENSION = "EXTENSION"

def movement_state(t):
    """Returns FLEXION when sin(2π·0.3·t) > 0, EXTENSION otherwise."""
    return MovementState.FLEXION if float(sin(2 * pi * 0.3 * t)) > 0 else MovementState.EXTENSION

t0 = 1.0
r_max = pi / 2
r = lambda t: r_max * (t > t0)
r = lambda t: r_max * sin(2 * pi * 0.5 * t)
# Square wave reference with 10% leading-edge delay per half-period:
# Active only after the first 10% of each FLEXION phase (0.1·π phase offset).
_ref_phase_delay = 0.2 * pi
r = lambda t: r_max * ((sin(2*pi*0.3*t) > 0)).astype(float)

# Extension MRAC reference: r_max during extension phases, 0 during flexion phases.
# No startup delay needed since the motor is pre-wound to r_max at t=0.
t_ext_delay = 0.2   # [s] kept for legacy use in movement logic
r_ext = lambda t: r_max * ((sin(2*pi*0.3*t) < 0)).astype(float)

"""
------------------------ Finger dynamics constants -----------------------
"""

phi1_0 = 0.0
phi2_0 = 0.0
phi3_0 = 0.0

simulation_speed = 1.0  # Playback speed multiplier (0.5 = half speed, 1.0 = real-time, etc.)

# ---- Link force configuration ----------------------------------------
# Attachment positions: distance from the proximal joint of each link [m]
# Allowed ranges:  0 <= force_s1 <= l1,  0 <= force_s2 <= l2,  0 <= force_s3 <= l3
force_s1 = l1 * 0.5   # link 1 – circle mounted at midpoint
force_s2 = l2 * 0.5   # link 2 – circle mounted at midpoint
force_s3 = l3 * 0.5   # link 3 – circle mounted at midpoint

# Circle radii mounted at each link midpoint [m]
# The pulling force is applied at the circumference of the circle,
# offset perpendicularly (radially outward, 90° CCW from link axis) from the midpoint.
r_circle1 = 0.010   # link 1 – circle radius [m]
r_circle2 = 0.010   # link 2 – circle radius [m]
r_circle3 = 0.010   # link 3 – circle radius [m]

# ---- Whiffle tree configuration -----------------------------------------
# One motor pulls a whiffle tree that distributes force to links 1, 2, and 3.
# wt_frac1 + wt_frac2 + wt_frac3 must equal 1.
wt_frac1 = 3/6   # fraction of motor force delivered to link 1
wt_frac2 = 2/6   # fraction of motor force delivered to link 2
wt_frac3 = 1/6   # fraction of motor force delivered to link 3

# Aim target fractions (0–1): the force on each link points FROM the attachment
# point TOWARDS a fractional position along the link below it.
#   0.0 → proximal joint of the link below
#   1.0 → distal joint of the link below
# Link 1 aims at the metacarpal (length l0), link 2 at link 1, link 3 at link 2.
aim_frac1 = 0.5   # link 1 → aims at aim_frac1 * l0 along the metacarpal
aim_frac2 = 0.5   # link 2 → aims at aim_frac2 * l1 along link 1
aim_frac3 = 0.5   # link 3 → aims at aim_frac3 * l2 along link 2

_force_s   = (force_s1, force_s2, force_s3)      
_force_r   = (r_circle1, r_circle2, r_circle3)   
_force_aim = (aim_frac1, aim_frac2, aim_frac3)
_force_aim_ext = (0.5, 0.5, 0.5)   # extension targets at link midpoints — same as flexion

# ---- Baumgarte constraint stabilization gains ----------------------------
# These correct numerical drift in the holonomic cable-length constraint
# r_spindle*theta + L_finger(phi) = C_cable.
# Setting both to zero recovers the original (drift-prone) formulation.
# A value of ~50 damps constraint errors in roughly 1/50 seconds.
alpha_B = 50.0   # velocity-level correction gain
beta_B  = 50.0   # position-level correction gain

# ---- Cable friction ------------------------------------------------------
# Models viscous friction losses in the cable routing (pulleys, guides, etc.).
# The motor "sees" the full cable tension; the tension *delivered* to the finger
# is reduced by  F_fric = cable_friction * |v_cable|  where
# v_cable = r_spindle * |omega| is the cable speed at the spindle.
# Set to 0.0 to disable (recovers the ideal frictionless model).
cable_friction = 0.0   # [N·s/m]  viscous cable friction coefficient

def _finger_cable_length(phi_1, phi_2, phi_3=0.0):
    """Geometric cable length on the finger side of the whiffle tree [m].
    Decreases as the finger flexes (cable is pulled taut). Flexion cable: 90° CCW offset."""
    a1 = phi_1
    a2 = phi_1 + phi_2
    a3 = phi_1 + phi_2 + phi_3
    MCP = np.array([l1*cos(a1), l1*sin(a1)])
    PIP = MCP + np.array([l2*cos(a2), l2*sin(a2)])
    att1 = np.array([force_s1*cos(a1) - r_circle1*sin(a1),
                     force_s1*sin(a1) + r_circle1*cos(a1)])
    att2 = MCP + np.array([force_s2*cos(a2) - r_circle2*sin(a2),
                           force_s2*sin(a2) + r_circle2*cos(a2)])
    att3 = PIP + np.array([force_s3*cos(a3) - r_circle3*sin(a3),
                           force_s3*sin(a3) + r_circle3*cos(a3)])
    target1 = np.array([-_force_aim[0] * l0, 0.0])
    target2 = _force_aim[1] * l1 * np.array([cos(a1), sin(a1)])
    target3 = MCP + _force_aim[2] * l2 * np.array([cos(a2), sin(a2)])
    return (wt_frac1 * np.linalg.norm(target1 - att1)
            + wt_frac2 * np.linalg.norm(target2 - att2)
            + wt_frac3 * np.linalg.norm(target3 - att3))


def _finger_cable_length_ext(phi_1, phi_2, phi_3=0.0):
    """Geometric cable length for the extension cable [m].
    Extension cable: attachment offset 90° CW from link axis (+sin, -cos).
    Same target points as flexion cable."""
    a1 = phi_1
    a2 = phi_1 + phi_2
    a3 = phi_1 + phi_2 + phi_3
    MCP = np.array([l1*cos(a1), l1*sin(a1)])
    PIP = MCP + np.array([l2*cos(a2), l2*sin(a2)])
    # Extension attachment: 90° CW offset (+sin, -cos)
    att_e1 = np.array([force_s1*cos(a1) + r_circle1*sin(a1),
                       force_s1*sin(a1) - r_circle1*cos(a1)])
    att_e2 = MCP + np.array([force_s2*cos(a2) + r_circle2*sin(a2),
                              force_s2*sin(a2) - r_circle2*cos(a2)])
    att_e3 = PIP + np.array([force_s3*cos(a3) + r_circle3*sin(a3),
                              force_s3*sin(a3) - r_circle3*cos(a3)])
    # Extension targets: target1 static (below base), targets 2/3 offset dorsally
    # (down relative to proximal link) by d_e from each joint.
    d_e = 0.01   # [m] dorsal offset magnitude; dorsal direction = [+sin, -cos]
    target1 = np.array([0.0, -d_e])
    target2 = MCP + d_e * np.array([ sin(a1), -cos(a1)])
    target3 = PIP + d_e * np.array([ sin(a2), -cos(a2)])
    return (wt_frac1 * np.linalg.norm(target1 - att_e1)
            + wt_frac2 * np.linalg.norm(target2 - att_e2)
            + wt_frac3 * np.linalg.norm(target3 - att_e3))


# Cable-length constant: r_spindle*theta + L_finger(phi) = C_cable (conserved)
C_cable = r_spindle * 0.0 + _finger_cable_length(phi1_0, phi2_0, phi3_0)
# Extension cable-length constant — extensor motor starts pre-wound to r_max,
# so C_cable_ext is computed at theta_e = r_max to ensure zero initial tension.
C_cable_ext = r_spindle * r_max + _finger_cable_length_ext(phi1_0, phi2_0, phi3_0)


def current_clamp(i_cmd):
    return float(np.clip(i_cmd, -Ia_stall, Ia_stall))


def control_law(x: np.ndarray, r_val: float, K: np.ndarray, L: float):
    i_cmd = -K @ x + L * r_val
    return current_clamp(i_cmd)


def no_load_control_law(x: np.ndarray, r_val: float):
    i_cmd = Kp * (r_val - x[0]) - Kd * x[1]
    return current_clamp(i_cmd)


def closed_loop_dynamics(t, z):
    """
    State:
    z = [theta_f, omega_f, theta_m_f, omega_m_f, K1_f, K2_f, L_f,        (0-6, flexion motor)
         theta_e, omega_e, theta_m_e, omega_m_e, K1_e, K2_e, L_e,        (7-13, extension motor)
         phi1, phi2, phi3, phi1_dot, phi2_dot, phi3_dot]                  (14-19, finger)

    Architecture:
      - Two MRAC outer loops: flexion and extension, each with cable constraint.
      - Flexion cable: 90° CCW attachment, pulls toward palmar targets → flexes finger.
      - Extension cable: 90° CW attachment, pulls toward same targets → extends finger.
      - Baumgarte stabilization on each constraint.
      - Unilateral cables: each goes slack independently.
    """
    # ---- Flexion motor state ----
    theta_f, omega_f = z[0], z[1]
    xm_f = z[2:4]
    K_f  = z[4:6]
    L_f  = z[6]

    # ---- Extension motor state ----
    theta_e, omega_e = z[7], z[8]
    xm_e = z[9:11]
    K_e  = z[11:13]
    L_e  = z[13]

    # ---- Finger state ----
    phi_1, phi_2, phi_3 = z[14:17]
    phi_1_dot, phi_2_dot, phi_3_dot = z[17:20]

    x_f = np.array([theta_f, omega_f])
    x_e = np.array([theta_e, omega_e])

    # ---- Movement state: determines which motor uses MRAC ----
    # FLEXION:   flexor  → MRAC with positive r_f,  extensor → PD with negative r_e
    # EXTENSION: extensor → MRAC with positive r_e, flexor   → PD with r_f = 0
    _state      = movement_state(t)
    _in_flexion = (_state == MovementState.FLEXION)
    _use_mrac_f = _in_flexion  and not should_use_pure_pd_control
    _use_mrac_e = (not _in_flexion) and (t > t_ext_delay) and not should_use_pure_pd_control

    r_f = float(r(t))          if _in_flexion else 0.0            # MRAC ref or PD hold at zero
    r_e = 0.0                  if _in_flexion else float(r_ext(t))  # PD hold at 0 or MRAC ref r_max
    e_f = x_f - xm_f
    e_e = x_e - xm_e

    # ---- Flexion current command ----
    if _use_mrac_f:
        i_cmd_f = control_law(x_f, r_f, K_f, L_f)
    else:
        i_cmd_f = no_load_control_law(x_f, r_f)

    # ---- Extension current command ----
    if _use_mrac_e:
        i_cmd_e = control_law(x_e, r_e, K_e, L_e)
    else:
        i_cmd_e = no_load_control_law(x_e, r_e)

    # ---- Cumulative joint angles ----
    a1 = phi_1
    a2 = phi_1 + phi_2
    a3 = phi_1 + phi_2 + phi_3
    MCP = np.array([l1*cos(a1), l1*sin(a1)])
    PIP = MCP + np.array([l2*cos(a2), l2*sin(a2)])

    # ---- Flexion cable attachment points (90° CCW: -sin, +cos) ----
    att1_f = np.array([force_s1*cos(a1) - r_circle1*sin(a1),
                       force_s1*sin(a1) + r_circle1*cos(a1)])
    att2_f = MCP + np.array([force_s2*cos(a2) - r_circle2*sin(a2),
                              force_s2*sin(a2) + r_circle2*cos(a2)])
    att3_f = PIP + np.array([force_s3*cos(a3) - r_circle3*sin(a3),
                              force_s3*sin(a3) + r_circle3*cos(a3)])

    # ---- Extension cable attachment points (90° CW: +sin, -cos) ----
    att1_e = np.array([force_s1*cos(a1) + r_circle1*sin(a1),
                       force_s1*sin(a1) - r_circle1*cos(a1)])
    att2_e = MCP + np.array([force_s2*cos(a2) + r_circle2*sin(a2),
                              force_s2*sin(a2) - r_circle2*cos(a2)])
    att3_e = PIP + np.array([force_s3*cos(a3) + r_circle3*sin(a3),
                              force_s3*sin(a3) - r_circle3*cos(a3)])

    # ---- Flexion target points (aim at midpoint of proximal link) ----
    target1_f = np.array([-_force_aim[0] * l0, 0.0])
    target2_f = _force_aim[1] * l1 * np.array([cos(a1), sin(a1)])
    target3_f = MCP + _force_aim[2] * l2 * np.array([cos(a2), sin(a2)])

    targety_e = -0.01  # extension targets below the finger (negative y)

    d_e = abs(targety_e)   # dorsal offset magnitude

    # ---- Extension target points: directly below each joint on the dorsal side ----
    target1_e = np.array([0.0, targety_e])
    target2_e = MCP + d_e * np.array([ sin(a1), -cos(a1)])
    target3_e = PIP + d_e * np.array([ sin(a2), -cos(a2)])

    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else np.zeros(2)

    # ---- Unit direction vectors (flexion) ----
    d1_f = _unit(target1_f - att1_f)
    d2_f = _unit(target2_f - att2_f)
    d3_f = _unit(target3_f - att3_f)

    # ---- Unit direction vectors (extension) ----
    d1_e = _unit(target1_e - att1_e)
    d2_e = _unit(target2_e - att2_e)
    d3_e = _unit(target3_e - att3_e)

    # ---- Positional Jacobians of attachment points ----
    # Flexion (90° CCW: r_circle term is [-sin, cos])
    J1_f = np.array([
        [-(force_s1*sin(a1) + r_circle1*cos(a1)), 0., 0.],
        [ force_s1*cos(a1) - r_circle1*sin(a1),   0., 0.],
    ])
    J2_f = np.array([
        [-(l1*sin(a1) + force_s2*sin(a2) + r_circle2*cos(a2)), -(force_s2*sin(a2) + r_circle2*cos(a2)), 0.],
        [ l1*cos(a1) + force_s2*cos(a2) - r_circle2*sin(a2),    force_s2*cos(a2) - r_circle2*sin(a2),   0.],
    ])
    J3_f = np.array([
        [-(l1*sin(a1) + l2*sin(a2) + force_s3*sin(a3) + r_circle3*cos(a3)),
         -(l2*sin(a2) + force_s3*sin(a3) + r_circle3*cos(a3)),
         -(force_s3*sin(a3) + r_circle3*cos(a3))],
        [ l1*cos(a1) + l2*cos(a2) + force_s3*cos(a3) - r_circle3*sin(a3),
          l2*cos(a2) + force_s3*cos(a3) - r_circle3*sin(a3),
          force_s3*cos(a3) - r_circle3*sin(a3)],
    ])

    # Extension (90° CW: r_circle term is [+sin, -cos])
    J1_e = np.array([
        [-(force_s1*sin(a1) - r_circle1*cos(a1)), 0., 0.],
        [ force_s1*cos(a1) + r_circle1*sin(a1),   0., 0.],
    ])
    J2_e = np.array([
        [-(l1*sin(a1) + force_s2*sin(a2) - r_circle2*cos(a2)), -(force_s2*sin(a2) - r_circle2*cos(a2)), 0.],
        [ l1*cos(a1) + force_s2*cos(a2) + r_circle2*sin(a2),    force_s2*cos(a2) + r_circle2*sin(a2),   0.],
    ])
    J3_e = np.array([
        [-(l1*sin(a1) + l2*sin(a2) + force_s3*sin(a3) - r_circle3*cos(a3)),
         -(l2*sin(a2) + force_s3*sin(a3) - r_circle3*cos(a3)),
         -(force_s3*sin(a3) - r_circle3*cos(a3))],
        [ l1*cos(a1) + l2*cos(a2) + force_s3*cos(a3) + r_circle3*sin(a3),
          l2*cos(a2) + force_s3*cos(a3) + r_circle3*sin(a3),
          force_s3*cos(a3) + r_circle3*sin(a3)],
    ])

    # ---- Moving-target Jacobians (flexion) ----
    J_target2_f = np.array([
        [-_force_aim[1] * l1 * sin(a1), 0., 0.],
        [ _force_aim[1] * l1 * cos(a1), 0., 0.],
    ])
    J_target3_f = np.array([
        [-(l1*sin(a1) + _force_aim[2]*l2*sin(a2)), -_force_aim[2]*l2*sin(a2), 0.],
        [ l1*cos(a1) + _force_aim[2]*l2*cos(a2),    _force_aim[2]*l2*cos(a2), 0.],
    ])

    # ---- Moving-target Jacobians (extension) ----
    # tgt1_e = [0, -d_e]  → constant → J = 0
    J_target1_e = np.zeros((2, 3))
    # tgt2_e = MCP + d_e*[sin(a1), -cos(a1)]  →  J = J_MCP + d_e * d/da1([sin, -cos])
    J_target2_e = np.array([
        [-l1*sin(a1) + d_e*cos(a1), 0., 0.],
        [ l1*cos(a1) + d_e*sin(a1), 0., 0.],
    ])
    # tgt3_e = PIP + d_e*[sin(a2), -cos(a2)]  →  a2 = phi1+phi2
    J_target3_e = np.array([
        [-l1*sin(a1) - l2*sin(a2) + d_e*cos(a2), -l2*sin(a2) + d_e*cos(a2), 0.],
        [ l1*cos(a1) + l2*cos(a2) + d_e*sin(a2),  l2*cos(a2) + d_e*sin(a2), 0.],
    ])

    # ---- Cable Jacobians ----
    J_cable_f = (wt_frac1 * (d1_f @ J1_f)
               + wt_frac2 * (d2_f @ (J2_f - J_target2_f))
               + wt_frac3 * (d3_f @ (J3_f - J_target3_f)))  # shape (3,)

    J_cable_e = (wt_frac1 * (d1_e @ (J1_e - J_target1_e))
               + wt_frac2 * (d2_e @ (J2_e - J_target2_e))
               + wt_frac3 * (d3_e @ (J3_e - J_target3_e)))  # shape (3,)

    # ---- Finger dynamics matrices ----
    M_mat = M(phi_1, phi_2, phi_3)
    C_mat = C(phi_1, phi_2, phi_3, phi_1_dot, phi_2_dot, phi_3_dot)
    tau_k = Tau_K(phi_1, phi_2, phi_3)
    tau_b = Tau_B(phi_1_dot, phi_2_dot, phi_3_dot)
    q_dot = np.array([phi_1_dot, phi_2_dot, phi_3_dot])
    tau_passive = -C_mat @ q_dot - tau_k - tau_b
    # Use np.linalg.solve instead of explicit inverse: faster and more numerically stable.
    # Pre-solve M^-1 against the three vectors we need.
    Minv_Jf  = np.linalg.solve(M_mat, J_cable_f)
    Minv_Je  = np.linalg.solve(M_mat, J_cable_e)
    Minv_tau = np.linalg.solve(M_mat, tau_passive)

    # ---- Coupled 2×2 constraint solve (Baumgarte) ----
    # Differentiating each constraint g_i = 0 twice and substituting both motor
    # EOMs plus the finger EOM (which sees forces from BOTH cables) yields:
    #
    #   [H_ff  H_fe] [T_f]   [rhs_f]
    #   [H_ef  H_ee] [T_e] = [rhs_e]
    #
    # where H_ff = r²/Jm + Jf·M⁻¹·Jf,  H_fe = Jf·M⁻¹·Je  (cross-coupling term)
    #       H_ef = Je·M⁻¹·Jf,           H_ee = r²/Jm + Je·M⁻¹·Je
    # The cross terms prevent the two constraints from fighting each other and
    # causing the solver to stall.
    # After solving, apply the unilateral constraint (cable can only pull):
    #   if either tension < 0, set it to 0 and re-solve the reduced 1×1 system.
    H_ff = r_spindle**2 / Jm + float(J_cable_f @ Minv_Jf)
    H_fe = float(J_cable_f @ Minv_Je)
    H_ef = float(J_cable_e @ Minv_Jf)
    H_ee = r_spindle**2 / Jm + float(J_cable_e @ Minv_Je)

    L_finger_f = (wt_frac1 * np.linalg.norm(target1_f - att1_f)
                + wt_frac2 * np.linalg.norm(target2_f - att2_f)
                + wt_frac3 * np.linalg.norm(target3_f - att3_f))
    g_pos_f = r_spindle * theta_f + L_finger_f - C_cable
    g_vel_f = r_spindle * omega_f - float(J_cable_f @ q_dot)

    L_finger_e = (wt_frac1 * np.linalg.norm(target1_e - att1_e)
                + wt_frac2 * np.linalg.norm(target2_e - att2_e)
                + wt_frac3 * np.linalg.norm(target3_e - att3_e))
    g_pos_e = r_spindle * theta_e + L_finger_e - C_cable_ext
    g_vel_e = r_spindle * omega_e - float(J_cable_e @ q_dot)

    rhs_f = (r_spindle * (Kt * i_cmd_f - Bm * omega_f) / Jm
             - float(J_cable_f @ Minv_tau)
             + 2.0 * alpha_B * g_vel_f + beta_B**2 * g_pos_f)
    rhs_e = (r_spindle * (Kt * i_cmd_e - Bm * omega_e) / Jm
             - float(J_cable_e @ Minv_tau)
             + 2.0 * alpha_B * g_vel_e + beta_B**2 * g_pos_e)

    # Solve coupled 2×2 system
    _det = H_ff * H_ee - H_fe * H_ef
    T_f_coupled = (H_ee * rhs_f - H_fe * rhs_e) / _det
    T_e_coupled = (H_ff * rhs_e - H_ef * rhs_f) / _det

    if T_f_coupled >= 0.0 and T_e_coupled >= 0.0:
        # Both taut: use the coupled solution
        T_cable_f_raw = T_f_coupled
        T_cable_e_raw = T_e_coupled
    elif T_f_coupled < 0.0 and T_e_coupled >= 0.0:
        # Flexion slack: solve for extension alone (T_f = 0 → reduced rhs)
        T_cable_f_raw = 0.0
        T_cable_e_raw = rhs_e / H_ee   # T_f=0 so cross-term vanishes
    elif T_f_coupled >= 0.0 and T_e_coupled < 0.0:
        # Extension slack: solve for flexion alone
        T_cable_f_raw = rhs_f / H_ff
        T_cable_e_raw = 0.0
    else:
        # Both slack
        T_cable_f_raw = 0.0
        T_cable_e_raw = 0.0

    T_cable_f = max(0.0, T_cable_f_raw)
    T_cable_e = max(0.0, T_cable_e_raw)

    # ---- Cable friction ----
    v_cable_f = r_spindle * abs(omega_f)
    v_cable_e = r_spindle * abs(omega_e)
    T_finger_f = max(0.0, T_cable_f - cable_friction * v_cable_f)
    T_finger_e = max(0.0, T_cable_e - cable_friction * v_cable_e)

    # ---- Finger EOM (both cables contribute) ----
    # Reuse already-solved Minv_Jf / Minv_Je / Minv_tau to avoid extra solves.
    q_ddot = Minv_tau + Minv_Jf * T_finger_f + Minv_Je * T_finger_e

    # ---- Motor EOMs ----
    dtheta_f = omega_f
    domega_f = (Kt * i_cmd_f - Bm * omega_f - r_spindle * T_cable_f) / Jm
    dtheta_e = omega_e
    domega_e = (Kt * i_cmd_e - Bm * omega_e - r_spindle * T_cable_e) / Jm

    # ---- Reference models ----
    dxm_f = A_m @ xm_f + B_m.flatten() * r_f
    dxm_e = A_m @ xm_e + B_m.flatten() * r_e

    # ---- MRAC adaptive laws ----
    if should_use_pure_pd_control:
        dK_f = np.zeros_like(K_f)
        dL_f = 0.0
        dK_e = np.zeros_like(K_e)
        dL_e = 0.0
    else:
        # Flexion adaptation (only while flexor uses MRAC and cable is taut)
        if _use_mrac_f and T_cable_f_raw >= 0:
            eps_f = (B_m.T @ P @ e_f.reshape(-1, 1)).item()
            dK_f  = Gamma_K_f @ x_f * eps_f - sigma_mod_f * Gamma_K_f @ K_f
            dL_f  = -gamma_L_f * r_f * eps_f - sigma_mod_f * gamma_L_f * L_f
        else:
            dK_f = np.zeros_like(K_f)
            dL_f = 0.0

        # Extension adaptation (only while extensor uses MRAC and cable is taut)
        if _use_mrac_e and T_cable_e_raw >= 0:
            eps_e = (B_m.T @ P @ e_e.reshape(-1, 1)).item()
            dK_e  = Gamma_K_e @ x_e * eps_e - sigma_mod_e * Gamma_K_e @ K_e
            dL_e  = -gamma_L_e * r_e * eps_e - sigma_mod_e * gamma_L_e * L_e
        else:
            dK_e = np.zeros_like(K_e)
            dL_e = 0.0

    # ---- Record cable tensions for post-processing (avoids re-running ODE) ----
    _cable_tension_record.append((t, T_cable_f, T_cable_e))

    return np.hstack([
        [dtheta_f, domega_f], dxm_f, dK_f, [dL_f],  # flexion motor  (7)
        [dtheta_e, domega_e], dxm_e, dK_e, [dL_e],  # extension motor (7)
        q_dot, q_ddot                                 # finger (6)
    ])


# Side-channel list populated during integration; cleared before each solve_ivp call.
_cable_tension_record = []


def main():
    """
    Simulation parameters
    """
    T = 10.0
    N = 1000
    t_eval = np.linspace(0, T, N)

    # [theta_f, omega_f, theta_m_f, omega_m_f, K1_f, K2_f, L_f,
    #  theta_e, omega_e, theta_m_e, omega_m_e, K1_e, K2_e, L_e,
    #  phi1, phi2, phi3, phi1_dot, phi2_dot, phi3_dot]
    z0 = [
        0.0, 0.0,                # flexion plant: theta, omega
        0.0, 0.0,                # flexion reference model: theta_m, omega_m
        K_0[0], K_0[1],          # flexion adaptive K
        L_0,                     # flexion adaptive L
        r_max, 0.0,              # extension plant: theta, omega (pre-wound to r_max)
        r_max, 0.0,              # extension reference model: theta_m, omega_m (matches plant)
        K_0[0], K_0[1],          # extension adaptive K
        L_0,                     # extension adaptive L
        phi1_0, phi2_0, phi3_0,  # finger initial angles
        0.0, 0.0, 0.0            # finger initial angular velocities
    ]

    save_folder = "adaptive_control/figures/with_finger_dynamics"
    filename = "mrac_pd_two_motors"  # Base filename for saved figures and animations
    should_save_animation = True  # Set to True to save the animation as a GIF file
    should_show_plots = False  # Set to False to skip showing plots (useful when only saving the animation)
    os.makedirs(save_folder, exist_ok=True)

    _cable_tension_record.clear()
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
            rtol=1e-4,
            atol=1e-6
        )

    # Extract results
    theta_f  = sol.y[0]
    omega_f  = sol.y[1]
    theta_mf = sol.y[2]
    omega_mf = sol.y[3]
    K1_f     = sol.y[4]
    K2_f     = sol.y[5]
    L_f      = sol.y[6]
    theta_e  = sol.y[7]
    omega_e  = sol.y[8]
    theta_me = sol.y[9]
    omega_me = sol.y[10]
    K1_e     = sol.y[11]
    K2_e     = sol.y[12]
    L_e      = sol.y[13]
    phi      = sol.y[14:17]
    phi_dot  = sol.y[17:20]

    print("Flexion adaptive gains at final time:")
    print(f"K_f = [{float(K1_f[-1])}, {float(K2_f[-1])}]")
    print(f"L_f = {L_f[-1]}")
    print("Extension adaptive gains at final time:")
    print(f"K_e = [{float(K1_e[-1])}, {float(K2_e[-1])}]")
    print(f"L_e = {L_e[-1]}")

    r_all   = r(sol.t)
    x_f_all = sol.y[0:2]
    x_e_all = sol.y[7:9]

    # Reconstruct the state-aware references actually seen by each motor
    _in_fl_arr     = np.array([movement_state(ti) == MovementState.FLEXION for ti in sol.t])
    _use_mrac_e_arr = (~_in_fl_arr) & (sol.t > t_ext_delay)
    # Flexion motor: positive ref during FLEXION, 0 during EXTENSION
    r_f_all = np.where(_in_fl_arr, r_all, 0.0)
    # Extension motor: 0 during FLEXION, r_ext(t) during EXTENSION (matches ODE logic)
    r_e_all = np.where(_in_fl_arr, 0.0, r_ext(sol.t))

    if should_use_pure_pd_control:
        i_cmd_f = np.clip(Kp * (r_f_all - x_f_all[0]) - Kd * x_f_all[1], -Ia_stall, Ia_stall)
        i_cmd_e = np.clip(Kp * (r_e_all - x_e_all[0]) - Kd * x_e_all[1], -Ia_stall, Ia_stall)
    else:
        # Flexion: MRAC during FLEXION phase, PD (r=0) during EXTENSION phase
        i_cmd_f_mrac = np.clip(-(K1_f * x_f_all[0] + K2_f * x_f_all[1]) + L_f * r_f_all, -Ia_stall, Ia_stall)
        i_cmd_f_pd   = np.clip(Kp * (r_f_all - x_f_all[0]) - Kd * x_f_all[1],            -Ia_stall, Ia_stall)
        i_cmd_f      = np.where(_in_fl_arr, i_cmd_f_mrac, i_cmd_f_pd)
        # Extension: PD (hold at 0) during FLEXION phase, MRAC during EXTENSION phase
        i_cmd_e_mrac = np.clip(-(K1_e * x_e_all[0] + K2_e * x_e_all[1]) + L_e * r_e_all, -Ia_stall, Ia_stall)
        i_cmd_e_pd   = np.clip(Kp * (r_e_all - x_e_all[0]) - Kd * x_e_all[1],                 -Ia_stall, Ia_stall)
        i_cmd_e      = np.where(_use_mrac_e_arr, i_cmd_e_mrac, i_cmd_e_pd)

    # Plot results
    _n_subplots = 4 if should_use_pure_pd_control else 5
    _fig_h      = 16 if should_use_pure_pd_control else 20
    fig, axes = plt.subplots(_n_subplots, 1, figsize=(12, _fig_h), sharex=True)

    def setup_axis(ax, ylabel, title=None):
        if title:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    # Flexion motor angle tracking
    axes[0].plot(sol.t, theta_mf, '--', label=r'$\theta_{m,f}$', color='tab:blue')
    axes[0].plot(sol.t, theta_f,  label=r'$\theta_f$', color='tab:blue', alpha=0.7)
    axes[0].plot(sol.t, r_f_all, 'r--', alpha=0.5, label=r'$r_f$')
    axes[0].legend()
    setup_axis(axes[0], 'Angle (rad)', title='Flexion Motor Angle Tracking')

    # Extension motor angle tracking
    axes[1].plot(sol.t, r_e_all, color='red', linestyle=':', alpha=0.5, label=r'$r_e$')
    axes[1].plot(sol.t, theta_me, '--', label=r'$\theta_{m,e}$', color='tab:orange')
    axes[1].plot(sol.t, theta_e,  label=r'$\theta_e$', color='tab:orange', alpha=0.7)
    axes[1].legend()
    setup_axis(axes[1], 'Angle (rad)', title='Extension Motor Angle Tracking')

    # Finger joint angles
    axes[2].axhline(phi1_eq, linestyle='--', label=r'$\phi_{1,\mathrm{eq}}$', color='tab:blue')
    axes[2].plot(sol.t, phi[0], label=r'$\phi_1$', color='tab:blue')
    axes[2].axhline(phi2_eq, linestyle='--', label=r'$\phi_{2,\mathrm{eq}}$', color='tab:orange')
    axes[2].plot(sol.t, phi[1], label=r'$\phi_2$', color='tab:orange')
    axes[2].axhline(phi3_eq, linestyle='--', label=r'$\phi_{3,\mathrm{eq}}$', color='tab:green')
    axes[2].plot(sol.t, phi[2], label=r'$\phi_3$', color='tab:green')
    axes[2].legend()
    setup_axis(axes[2], 'Relative Angle (rad)', title='Finger Joint Angles')

    # Adaptive gains (skipped for pure PD mode)
    if not should_use_pure_pd_control:
        axes[3].plot(sol.t, K1_f, label=r'$K_{1,f}$')
        axes[3].plot(sol.t, K2_f, label=r'$K_{2,f}$')
        axes[3].plot(sol.t, L_f,  label=r'$L_f$')
        axes[3].plot(sol.t, K1_e, label=r'$K_{1,e}$', linestyle='--')
        axes[3].plot(sol.t, K2_e, label=r'$K_{2,e}$', linestyle='--')
        axes[3].plot(sol.t, L_e,  label=r'$L_e$', linestyle='--')
        axes[3].legend(ncol=2)
        setup_axis(axes[3], 'Gain', title='Adaptive Gains')

    # Motor current commands — both on one subplot
    _ax_i = 3 if should_use_pure_pd_control else 4
    axes[_ax_i].plot(sol.t, i_cmd_f, label=r'$i_{\mathrm{cmd},f}$')
    axes[_ax_i].plot(sol.t, i_cmd_e, label=r'$i_{\mathrm{cmd},e}$', color='darkorange')
    axes[_ax_i].axhline( Ia_stall, color='r', linestyle=':', linewidth=1.2, label=r'$\pm I_\mathrm{stall}$')
    axes[_ax_i].axhline(-Ia_stall, color='r', linestyle=':', linewidth=1.2)
    axes[_ax_i].legend()
    setup_axis(axes[_ax_i], 'Current (A)', title='Motor Current Commands')
    axes[_ax_i].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(f"{save_folder}/{filename}.png", dpi=300)

    # ---- Interpolate cable tensions recorded during integration ----
    # This avoids re-running the full cable geometry for every t_eval point.
    _t_rec  = np.array([_r[0] for _r in _cable_tension_record])
    _Tf_rec = np.array([_r[1] for _r in _cable_tension_record])
    _Te_rec = np.array([_r[2] for _r in _cable_tension_record])
    T_cable_f_all = np.interp(sol.t, _t_rec, _Tf_rec)
    T_cable_e_all = np.interp(sol.t, _t_rec, _Te_rec)

    # ---- Add red slack shading to all axes ----
    _slack_f = T_cable_f_all == 0.0
    _slack_e = T_cable_e_all == 0.0
    _t = sol.t
    def _add_shading(ax, mask, color, alpha=0.12):
        """Shade regions where mask is True."""
        in_span = False
        t_start = None
        for k in range(len(_t)):
            if mask[k] and not in_span:
                t_start = _t[k]; in_span = True
            elif not mask[k] and in_span:
                ax.axvspan(t_start, _t[k], color=color, alpha=alpha)
                in_span = False
        if in_span:
            ax.axvspan(t_start, _t[-1], color=color, alpha=alpha)

    for _ax in axes:
        _add_shading(_ax, _slack_f, color='red',  alpha=0.12)   # flexion slack = red
        _add_shading(_ax, _slack_e, color='blue', alpha=0.10)   # extension slack = blue

    # ---- Interpolating callables — per-link cable forces [N] ----
    # Animation shows net flexion force (flexion minus extension contribution)
    # per link. Both tensions are positive-only; sign is embedded in J_cable direction.
    _t_arr = sol.t
    # Flexion per-link forces
    _F1_f = wt_frac1 * T_cable_f_all
    _F2_f = wt_frac2 * T_cable_f_all
    _F3_f = wt_frac3 * T_cable_f_all
    # Extension per-link forces (shown as negative so arrow direction reflects extension)
    _F1_e = wt_frac1 * T_cable_e_all
    _F2_e = wt_frac2 * T_cable_e_all
    _F3_e = wt_frac3 * T_cable_e_all
    # Net per-link force (for scaling arrows)
    _F1_net = _F1_f - _F1_e
    _F2_net = _F2_f - _F2_e
    _F3_net = _F3_f - _F3_e
    # Flexion callables (palmar arrows, red/orange)
    _lf_mag_anim = (
        lambda t, s: float(np.interp(t, _t_arr, _F1_f)),
        lambda t, s: float(np.interp(t, _t_arr, _F2_f)),
        lambda t, s: float(np.interp(t, _t_arr, _F3_f)),
    )
    # Extension callables (dorsal arrows, blue)
    _lf_mag_anim_ext = (
        lambda t, s: float(np.interp(t, _t_arr, _F1_e)),
        lambda t, s: float(np.interp(t, _t_arr, _F2_e)),
        lambda t, s: float(np.interp(t, _t_arr, _F3_e)),
    )
    # Use the larger of the two cable maxima as the common arrow scale
    _force_scale = float(np.max(np.abs([_F1_f, _F2_f, _F3_f, _F1_e, _F2_e, _F3_e])))
    if _force_scale < 1e-12:
        _force_scale = 1.0

    # ---- Finger animation -----------------------------------------------
    sol_finger = SimpleNamespace(
        t=sol.t,
        y=sol.y[14:20]   # rows: phi1, phi2, phi3, phi1_dot, phi2_dot, phi3_dot
    )

    if should_show_plots:
        _ = animate_finger_simulation(
            sol_finger, l1, l2, l3,
            speed=simulation_speed,
            link_force_s=_force_s,
            link_force_mag=_lf_mag_anim,
            link_force_r=_force_r,
            aim_frac=_force_aim,
            aim_frac_ext=_force_aim_ext,
            l0=l0,
            force_scale=_force_scale,
            link_force_mag_ext=_lf_mag_anim_ext,
            eq_angles=(phi1_eq, phi2_eq, phi3_eq),
        )

    if should_save_animation:
        anim_filepath = f"{save_folder}/{filename}.gif"
        print(f"Saving animation to {anim_filepath} ...")
        anim_save = animate_finger_simulation(
            sol_finger, l1, l2, l3,
            speed=simulation_speed, save_fps=30,
            link_force_s=_force_s,
            link_force_mag=_lf_mag_anim,
            link_force_r=_force_r,
            aim_frac=_force_aim,
            aim_frac_ext=_force_aim_ext,
            l0=l0,
            force_scale=_force_scale,
            link_force_mag_ext=_lf_mag_anim_ext,
            eq_angles=(phi1_eq, phi2_eq, phi3_eq),
        )
        anim_save.save(anim_filepath, writer='pillow', fps=30, dpi=150)
        plt.close(anim_save._fig)
        print("Animation saved.")

    if should_show_plots:
        plt.show()

if __name__ == "__main__":
    main()