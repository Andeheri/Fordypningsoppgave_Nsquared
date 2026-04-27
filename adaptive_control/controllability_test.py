
import numpy as np
from numpy import pi
import sympy as sp

"""
Motor parameters
"""
# V_supply = 12.0  # V (Supply voltage)
# g = 9.81  # m/s^2 (Gravitational acceleration)
# # Information from datasheet: https://www.pololu.com/file/0J1829/pololu-25d-metal-gearmotors.pdf, page 3
# Ia_stall = 4.9  # A
# tau_stall = 220 * g / 1000 # Nm (Stall torque at 12 V)

# Ia_no_load = 0.2  # A (No-load current at 12 V)
# theta_dot_no_load = 130 * 2 * pi / 60  # rad/s (No-load speed at 12 V)

# Kt = tau_stall / Ia_stall # Nm/A (Torque constant) ≈ 0.44 Nm/A
# Ra = V_supply / Ia_stall # Ω (Armature resistance) ≈ 2.44 Ω (Measured from stall current at 12 V)
# Bm = Kt * Ia_no_load / theta_dot_no_load  # Nm*s (Rotor friction coefficient)
# Kb = (V_supply - Ra * Ia_no_load) / theta_dot_no_load   # V*s/rad (Back EMF constant)

# # Unkowns
# Jm = 0.093  # kg*m^2 (Rotor moment of inertia)
# La = 0.006  # H (Armature inductance)

# # Mass spring damper load
# J = 0.5
# c = 0.2
# k = 1.0

k, c, J = sp.symbols('k c J', real=True, positive=True)
Jm, Bm = sp.symbols('Jm Bm', real=True, positive=True)
Kb, Ra, La, Kt = sp.symbols('Kb Ra La Kt', real=True, positive=True)

"""
State-space realisation of system dynamics (for simulation):
"""
A = sp.Matrix([
    [0, 1, 0],
    [-k/(Jm + J), - (Bm + c)/(Jm + J), Kt/(Jm + J)],
    [0, -Kb/La, -Ra/La]
])

B = sp.Matrix([
    [0],
    [0],
    [1/La]
])

controllability_matrix = sp.Matrix.hstack(B, A @ B, A @ A @ B)
controllability_matrix = controllability_matrix.applyfunc(lambda x: sp.simplify(x))
rank_of_controllability_matrix = controllability_matrix.rank()
print("Controllability matrix:")
sp.pprint(controllability_matrix, use_unicode=True)
print(f"Rank of controllability matrix: {rank_of_controllability_matrix}")