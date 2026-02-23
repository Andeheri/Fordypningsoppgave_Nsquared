
import sympy as sp
import time
from tqdm import tqdm

"""-------------------- Symbols -----------------------"""
t = sp.Symbol('t', real=True)
m1, m2, m3 = sp.symbols('m1 m2 m3', positive=True, real=True)  # Masses
r1, r2, r3 = sp.symbols('r1 r2 r3', positive=True, real=True)  # Radius of links
l1, l2, l3 = sp.symbols('l1 l2 l3', positive=True, real=True)  # Constants

k1, k2, k3 = sp.symbols('k1 k2 k3', positive=True, real=True)  # Spring constants
b1, b2, b3 = sp.symbols('b1 b2 b3', positive=True, real=True)  # Damping coefficients

theta1, theta2, theta3 = sp.symbols('theta_1 theta_2 theta_3', cls=sp.Function)
theta1_0, theta2_0, theta3_0 = sp.symbols('theta_1_0 theta_2_0 theta_3_0', real=True)  # Spring rest angles
θ1, θ2, θ3 = theta1(t), theta2(t), theta3(t)  # Functions of time
θ1_dot, θ2_dot, θ3_dot = sp.diff(θ1, t), sp.diff(θ2, t), sp.diff(θ3, t)

theta1_2, theta2_3, theta1_3 = sp.symbols('theta_1-2 theta_2-3 theta_1-3', cls=sp.Function)
θ1_2, θ2_3, θ1_3 = theta1_2(t), theta2_3(t), theta1_3(t)

θ1_2_dot, θ2_3_dot, θ1_3_dot = sp.diff(θ1_2, t), sp.diff(θ2_3, t), sp.diff(θ1_3, t)

"""-------------------- Helper functions -----------------------"""
def R_func(θ: sp.Symbol) -> sp.Matrix:   
    return sp.Matrix([[sp.cos(θ), -sp.sin(θ)],
                      [sp.sin(θ), sp.cos(θ)]]) 


def substitute_angles(expr: sp.Expr) -> sp.Expr:
    """
    Substitutes the total angles and their derivatives in the given expression.
    
    :param expr: The expression to substitute angles in
    :type expr: sp.Expr
    """
    expr = expr.subs(θ1 + θ2 + θ3, θ1_3)
    expr = expr.subs(θ1 + θ2, θ1_2)
    expr = expr.subs(θ2 + θ3, θ2_3)
    expr = expr.subs(θ1_dot + θ2_dot + θ3_dot, θ1_3_dot)
    expr = expr.subs(θ1_dot + θ2_dot, θ1_2_dot)
    expr = expr.subs(θ2_dot + θ3_dot, θ2_3_dot)
    return expr


def inertia_cylinder(m: sp.Symbol, r: sp.Symbol, l: sp.Symbol) -> sp.Expr:
    """
    Returns the moment of inertia for a cylinder rotating about its base.
    
    :param m: Mass of the cylinder
    :type m: sp.Symbol

    :param r: Radius of the cylinder
    :type r: sp.Symbol

    :param l: Length of the cylinder
    :type l: sp.Symbol
    """
    return sp.Rational(1, 12) * m * (3 * r**2 + 4 * l**2)


def homogeneous_transform(i: int, should_substitute=True) -> sp.Matrix:
    """
    Returns the HT for the i-th phalange, located at the end of the i-th phalange.
    
    :param i: I'th phalange (1, 2, or 3)
    :type i: int

    :param should_substitute: Whether to substitute the total angles and their derivatives
    :type should_substitute: bool
    """
    θ_tot = sum([θ1, θ2, θ3][:i])  # Total angle up to the i-th phalange
    
    R = R_func(θ_tot)
    d = sum(
        (
            R_func(sum([θ1, θ2, θ3][:j]))
            @ sp.Matrix([[[l1, l2, l3][j - 1]], [0]])
            for j in range(1, i + 1)
        ),
        sp.zeros(2, 1),
    )

    HT = R.row_join(d).col_join(sp.Matrix([[0, 0, 1]]))
    if should_substitute:
        HT = substitute_angles(HT)
    return sp.simplify(HT)


def diff_homogeneous_transform(i: int, should_substitute=True) -> sp.Matrix:
    """
    Returns the derivative of the HT for the i-th phalange with respect to the given variable.
    
    :param i: I'th phalange (1, 2, or 3)
    :type i: int

    :param should_substitute: Whether to substitute the total angles and their derivatives
    :type should_substitute: bool
    """
    dHT = sp.diff(homogeneous_transform(i, should_substitute = False), t)
    if should_substitute:   
        dHT = substitute_angles(dHT)
    return sp.simplify(dHT)


def r_CM(i: int, should_substitute=True) -> sp.Matrix:
    """
    Returns the position vector of the center of mass of the i-th phalange in the base frame.
    
    :param i: I'th phalange (1, 2, or 3)
    :type i: int
    """
    HT = homogeneous_transform(i, should_substitute=False)
    r_CM_local = sp.Matrix([[-[l1, l2, l3][i - 1] / 2],
                            [0],
                            [1]])
    r_CM_global = HT @ r_CM_local
    if should_substitute:   
        r_CM_global = substitute_angles(r_CM_global)
    return sp.simplify(r_CM_global[:2, 0])


def r_CM_dot(i: int, should_substitute=True) -> sp.Matrix:
    """
    Returns the velocity vector of the center o
    f mass of the i-th phalange in the base frame.
    
    :param i: I'th phalange (1, 2, or 3)
    :type i: int
    """
    dHT = diff_homogeneous_transform(i, should_substitute=False)
    r_CM_local = sp.Matrix([[-[l1, l2, l3][i - 1] / 2],
                                [0],
                                [1]])
    r_CM_global_dot = dHT @ r_CM_local
    if should_substitute:   
        r_CM_global_dot = substitute_angles(r_CM_global_dot)
    return sp.simplify(r_CM_global_dot[:2, 0])


def main():
    print("Computing variables for the robotic manuipulator ...")
    progress_bar = tqdm(total=4, desc="Calculating velocities of links")
    should_substitute = True

    J1 = inertia_cylinder(m1, r1, l1)
    J2 = inertia_cylinder(m2, r2, l2)
    J3 = inertia_cylinder(m3, r3, l3)

    r_CM_1_dot = r_CM_dot(1, should_substitute=False)
    r_CM_2_dot = r_CM_dot(2, should_substitute=False)
    r_CM_3_dot = r_CM_dot(3, should_substitute=False)

    J = sp.Matrix.diag(J1, J2, J3)
    M6 = sp.Matrix.diag(m1, m1, m2, m2, m3, m3)
    K = sp.Matrix.diag(k1, k2, k3)
    r_CM_dot_vec = sp.Matrix.vstack(r_CM_1_dot, r_CM_2_dot, r_CM_3_dot)
    theta_dot_vec = sp.Matrix([θ1_dot, θ2_dot, θ3_dot])
    theta_vec = sp.Matrix([θ1, θ2, θ3])
    progress_bar.update(1)
    progress_bar.set_description("Computing kinetic energy")
    # Kinetic energy
    T = (sp.Rational(1, 2) * (theta_dot_vec.T * J * theta_dot_vec + r_CM_dot_vec.T * M6 * r_CM_dot_vec))[0]
    # Unpack kinetic energy
    T = sp.simplify(T)
    progress_bar.update(1)
    progress_bar.set_description("Computing the generalized inertia matrix")

    M = sp.hessian(T, theta_dot_vec)   
    M = sp.simplify(M)                    
    M = substitute_angles(M)
    progress_bar.update(1)
    progress_bar.update(1)
    progress_bar.close()
    print("\nMass matrix M(q):")
    sp.pprint(M, use_unicode=True)



if __name__ == "__main__":
    main()