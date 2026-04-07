
"""
This script is based on the controller developed in chapter 3.13 - State-Space Identifiers
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numpy import sin, cos, pi
from scipy.integrate import solve_ivp
from tqdm import tqdm
from dynamics import *
from visualization.visualization_finger_simulation import animate_finger_simulation, plot_simulation_angles

"""
State-space representation of the 3-link planar finger dynamics
"""
def A(theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
    """
    State-space matrix A for the 3-link planar finger.
    
    A = [O3, I3;
         -M^-1*K, -M^-1*(C + Bf)]
    """
    A = np.block([[O3, I3],
                  [-np.linalg.solve(M(theta), Kf), -np.linalg.solve(M(theta), C(theta, theta_dot) + Bf)]])
    return A


def B(theta: np.ndarray) -> np.ndarray:
    """
    State-space matrix B for the 3-link planar finger.
    
    B = [O3;
         M^-1]
    """
    B = np.block([[O3],
                  [np.linalg.inv(M(theta))]])
    return B


def D(theta: np.ndarray) -> np.ndarray:
    """
    State-space matrix D for the 3-link planar finger.
    
    D = [O3;
         M^-1*Kf]
    """
    D = np.block([[O3],
                  [np.linalg.solve(M(theta), Kf)]])
    return D


def u(t: float, theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
    return np.array([1.0, 2.0, 3.0]) * sin(2 * pi * 0.5 * t)  # Example control input, can be replaced with a more complex controller


def state_space_dynamics(t: float, x: np.ndarray) -> np.ndarray:
    theta = x[:3]
    theta_dot = x[3:]
    tau = u(t, theta, theta_dot)
    theta_ddot = dynamics(theta, theta_dot, tau)
    return A(theta, theta_dot) @ x + B(theta) @ tau + D(theta) @ theta_spring_0


def ns2(x: np.ndarray, u: np.ndarray) -> float:
    return x.T @ x + u.T @ u

def ω_dot(A_m: np.ndarray, ω: np.ndarray, ϵ: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return A_m @ ω + ϵ * ns2(x, u)

def x_est_dot(A_m: np.ndarray, x_est: np.ndarray, x: np.ndarray, u: np.ndarray, A_hat: np.ndarray, B_hat: np.ndarray, D_hat: np.ndarray, theta_spring_0: np.ndarray) -> np.ndarray:
    return A_m @ (x_est - x) + A_hat @ x + B_hat @ u + D_hat @ theta_spring_0

def main():
    # Figure configuration
    folder_path = "figures/controller_performance/state_space_identifiers"
    file_name = "state_space_identifiers"
    file_path = os.path.join(folder_path, file_name)
    should_save_figures = True
    should_animate = False
    should_show_figures = False

    # Simulation parameters
    T = 5.0  # Total simulation time [s]
    N = 1000  # Number of evaluation points
    t_vals = np.linspace(0, T, N)
    dt = t_vals[1] - t_vals[0]

    # Initial conditions
    θ_0 = np.array([0.0, 0.0, 0.0])  # Initial joint angles [rad]
    θ_dot_0 = np.array([0.0, 0.0, 0.0])  # Initial joint angular velocities [rad/s]
    state0 = np.concatenate([θ_0, θ_dot_0])

    # Controller parameters
    γ1, γ2 = 10.0, 10.0  # Identifier learning rates
    A_m = np.diag([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]) # Stable matrix

    # Initial guess for the matrices
    M_0 = np.eye(3)
    C_0 = np.zeros((3, 3))
    Kf_0 = np.eye(3)
    Bf_0 = np.eye(3)

    M_0_inv = np.linalg.inv(M_0)
    A_hat_0 = np.block([[O3, I3],
                        [-M_0_inv @ Kf_0, -M_0_inv @ (C_0 + Bf_0)]])
    B_hat_0 = np.block([[O3],
                        [M_0_inv]])
    D_hat_0 = np.block([[O3],
                        [M_0_inv @ Kf_0]])
    
    # Simulation
    ω_0 = np.zeros((6, 1))  # Initial parameter estimation error
    ω = np.zeros((6, N))
    ω[:, 0] = ω_0.flatten()
    x = np.zeros((6, N))
    x[:, 0] = state0
    x_est = np.zeros((6, N))
    x_est[:, 0] = state0  # Initial state estimate

    # Print dynamics
    # print(f"M(θ_0) =\n{M(θ_0)}")
    # print(f"C(θ_0, θ_dot_0) =\n{C(θ_0, θ_dot_0)}")
    # print("At rest:")
    # print(f"M(θ_spring_0) =\n{M(theta_spring_0)}")
    # print(f"C(θ_spring_0, 0) =\n{C(theta_spring_0, np.zeros(3))}")

    def ode(t, state):
        x_cur     = state[:6]
        x_hat_cur = state[6:]
        theta     = x_cur[:3]
        theta_dot = x_cur[3:]
        tau       = u(t, theta, theta_dot)
        theta_ddot = dynamics(theta, theta_dot, tau)
        x_dot     = np.concatenate([theta_dot, theta_ddot])
        x_hat_dot = x_est_dot(A_m, x_hat_cur, x_cur, tau, A_hat_0, B_hat_0, D(theta), theta_spring_0)
        return np.concatenate([x_dot, x_hat_dot])

    augmented_state0 = np.concatenate([state0, state0])
    last_t = [0.0]
    with tqdm(total=T, desc="Simulating finger dynamics", unit="s", unit_scale=True) as pbar:
        def ode_with_progress(t, state):
            pbar.update(t - last_t[0])
            last_t[0] = t
            return ode(t, state)
        sol = solve_ivp(ode_with_progress, (0, T), augmented_state0, t_eval=t_vals,
                        method='Radau', rtol=1e-6, atol=1e-8)

    # # Visualization
    θ1, θ2, θ3 = sol.y[0], sol.y[1], sol.y[2]
    plot_simulation_angles(sol.t, θ1, θ2, θ3, theta1_0, theta2_0, theta3_0, file_path + ".png", should_save_figures)
    if should_animate:
        animate_finger_simulation(sol, l1, l2, l3, filepath=file_path + ".gif", should_save=should_save_figures, should_show=should_show_figures)


if __name__ == "__main__":
    main()