
import sympy as sp
import numpy as np

sp.init_printing(use_unicode=True)

# Define symbolic variables for joint angles and link lengths
theta_MCP, theta_PIP, theta_DIP = sp.symbols('theta_MCP theta_PIP theta_DIP')
l_PP, l_IP, l_DP = sp.symbols('l_PP l_IP l_DP')

# Define rotation matrices around the z-axis in 2D
R = lambda theta: sp.Matrix([[sp.cos(theta), -sp.sin(theta)],
                             [sp.sin(theta), sp.cos(theta)]])
# Define translation vectors in 2D
d = lambda x: sp.Matrix([x, 0])
# Define general Homogeneous transformation matrix in 2D
H = lambda R, d: sp.Matrix.vstack(
    sp.Matrix.hstack(R, R@d),
    sp.Matrix([[0, 0, 1]])
)

# Define individual transformation matrices for each joint
H_i_PP  = H(R(theta_MCP), d(l_PP))
H_PP_IP = H(R(theta_PIP), d(l_IP))
H_IP_DP = H(R(theta_DIP), d(l_DP))

# Calculate the overall transformation from the base to the fingertip
H_i_DP = H_i_PP * H_PP_IP * H_IP_DP
H_i_DP = sp.simplify(H_i_DP)

# Print results

def print_matrix(name, matrix):
    print(f"{name}:")
    sp.pprint(matrix, use_unicode=True)
    print()

print_matrix("H_i_DP", H_i_DP)

print("H_i_DP (numpy array):")
H_i_DP_np = np.array(H_i_DP.tolist(), dtype=object)
print(np.array2string(H_i_DP_np, separator=", "))