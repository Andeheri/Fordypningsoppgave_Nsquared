
import numpy as np
from scipy.integrate import solve_ivp

"""
Define motor parameters
"""

Jm = 0.093  # kg*m^2 (Rotor moment of inertia)
Bm = 0.008  # N*m*s (Rotor friction coefficient)
Kb = 0.6    # V*s/rad (Back EMF constant)
Kt = 0.7274 # Nm/A (Torque constant)
Ra = 0.6    # Ω (Armature resistance)
La = 0.006  # H (Armature inductance)

V_peak = 12.0  # V (Peak voltage applied to the motor)
pwm_frequency = 1000.0  # Hz (PWM frequency)
pwm_period = 1.0 / pwm_frequency  # s (PWM period)