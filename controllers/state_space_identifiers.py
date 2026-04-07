
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


def u(t: float, theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
    return np.array([1.0, 2.0, 3.0]) * sin(2 * pi * 0.5 * t)  # Example control input, can be replaced with a more complex controller


def main():
    # Figure configuration
    folder_path = "figures/controller_performance/state_space_identifiers"
    file_name = "without_controller"
    file_path = os.path.join(folder_path, file_name)
    should_save_figures = True
    should_animate = True
    should_show_figures = False

    # Simulation parameters
    T = 5.0  # Total simulation time [s]
    N = 2000  # Number of evaluation points

    # Initial conditions
    θ_0 = np.array([0.0, 0.0, 0.0])  # Initial joint angles [rad]
    θ_dot_0 = np.array([0.0, 0.0, 0.0])  # Initial joint angular velocities [rad/s]
    state0 = np.concatenate([θ_0, θ_dot_0])

    def ode(t, state):
        theta = state[:3]
        theta_dot = state[3:]
        tau = u(t, theta, theta_dot)
        theta_ddot = dynamics(theta, theta_dot, tau)
        return np.concatenate([theta_dot, theta_ddot])

    t_eval = np.linspace(0, T, N)
    last_t = [0.0]
    with tqdm(total=T, desc="Simulating finger dynamics", unit="s", unit_scale=True) as pbar:
        def ode_with_progress(t, state):
            pbar.update(t - last_t[0])
            last_t[0] = t
            return ode(t, state)
        sol = solve_ivp(ode_with_progress, (0, T), state0, t_eval=t_eval,
                        method='RK45', rtol=1e-8, atol=1e-10)

    # Visualization
    θ1, θ2, θ3 = sol.y[0], sol.y[1], sol.y[2]
    plot_simulation_angles(sol.t, θ1, θ2, θ3, theta1_0, theta2_0, theta3_0, file_path + ".png", should_save_figures)
    if should_animate:
        animate_finger_simulation(sol, l1, l2, l3, filepath=file_path + ".gif", should_save=should_save_figures, should_show=should_show_figures)


if __name__ == "__main__":
    main()