"""
Computes the symbolic mass matrix M(q) and exports it as a
copy-pasteable Python function to the clipboard.
"""

import sympy as sp
import subprocess
import sys
import time

t = sp.Symbol('t', real=True)
m1, m2, m3 = sp.symbols('m1 m2 m3', positive=True, real=True)
r1, r2, r3 = sp.symbols('r1 r2 r3', positive=True, real=True)
l1, l2, l3 = sp.symbols('l1 l2 l3', positive=True, real=True)

theta1, theta2, theta3 = sp.symbols('theta_1 theta_2 theta_3', cls=sp.Function)
θ1, θ2, θ3 = theta1(t), theta2(t), theta3(t)
θ1_dot, θ2_dot, θ3_dot = sp.diff(θ1, t), sp.diff(θ2, t), sp.diff(θ3, t)

theta1_2, theta2_3, theta1_3 = sp.symbols('theta_1-2 theta_2-3 theta_1-3', cls=sp.Function)
θ1_2, θ2_3, θ1_3 = theta1_2(t), theta2_3(t), theta1_3(t)
θ1_2_dot, θ2_3_dot, θ1_3_dot = sp.diff(θ1_2, t), sp.diff(θ2_3, t), sp.diff(θ1_3, t)


def R_func(θ):
    return sp.Matrix([[sp.cos(θ), -sp.sin(θ)],
                      [sp.sin(θ),  sp.cos(θ)]])


def substitute_angles(expr):
    expr = expr.subs(θ1 + θ2 + θ3, θ1_3)
    expr = expr.subs(θ1 + θ2, θ1_2)
    expr = expr.subs(θ2 + θ3, θ2_3)
    expr = expr.subs(θ1_dot + θ2_dot + θ3_dot, θ1_3_dot)
    expr = expr.subs(θ1_dot + θ2_dot, θ1_2_dot)
    expr = expr.subs(θ2_dot + θ3_dot, θ2_3_dot)
    return expr


def inertia_cylinder(m, r, l):
    return sp.Rational(1, 12) * m * (3 * r**2 + 4 * l**2)


def homogeneous_transform(i, should_substitute=True):
    θ_tot = sum([θ1, θ2, θ3][:i])
    R = R_func(θ_tot)
    d = sum(
        (R_func(sum([θ1, θ2, θ3][:j])) @ sp.Matrix([[[l1, l2, l3][j - 1]], [0]])
         for j in range(1, i + 1)),
        sp.zeros(2, 1),
    )
    HT = R.row_join(d).col_join(sp.Matrix([[0, 0, 1]]))
    if should_substitute:
        HT = substitute_angles(HT)
    return sp.simplify(HT)


def diff_homogeneous_transform(i, should_substitute=True):
    dHT = sp.diff(homogeneous_transform(i, should_substitute=False), t)
    if should_substitute:
        dHT = substitute_angles(dHT)
    return sp.simplify(dHT)


def r_CM_dot(i, should_substitute=True):
    dHT = diff_homogeneous_transform(i, should_substitute=False)
    r_CM_local = sp.Matrix([[- [l1, l2, l3][i - 1] / 2], [0], [1]])
    r_CM_global_dot = dHT @ r_CM_local
    if should_substitute:
        r_CM_global_dot = substitute_angles(r_CM_global_dot)
    return sp.simplify(r_CM_global_dot[:2, 0])


def main():
    print("Computing mass matrix M(q) ...")
    start = time.time()

    J1 = inertia_cylinder(m1, r1, l1)
    J2 = inertia_cylinder(m2, r2, l2)
    J3 = inertia_cylinder(m3, r3, l3)

    r1_dot = r_CM_dot(1, should_substitute=False)
    r2_dot = r_CM_dot(2, should_substitute=False)
    r3_dot = r_CM_dot(3, should_substitute=False)

    J_mat  = sp.Matrix.diag(J1, J2, J3)
    M6     = sp.Matrix.diag(m1, m1, m2, m2, m3, m3)
    r_dot_vec = sp.Matrix.vstack(r1_dot, r2_dot, r3_dot)
    theta_dot_vec = sp.Matrix([θ1_dot, θ2_dot, θ3_dot])

    T = (sp.Rational(1, 2) * (theta_dot_vec.T * J_mat * theta_dot_vec
                               + r_dot_vec.T * M6 * r_dot_vec))[0]
    T = sp.simplify(T)

    M = sp.hessian(T, [θ1_dot, θ2_dot, θ3_dot])
    M = sp.simplify(M)
    M = substitute_angles(M)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f} s")

    # ------------------------------------------------------------------
    # Rewrite M in terms of plain (non-time-dependent) symbols so that
    # it can be turned into a clean Python function.
    # ------------------------------------------------------------------
    q1, q2, q3 = sp.symbols('q1 q2 q3', real=True)

    # Substitute shorthand sums first, then individual angles
    M_plain = M
    M_plain = M_plain.subs(θ1_3, q1 + q2 + q3)
    M_plain = M_plain.subs(θ1_2, q1 + q2)
    M_plain = M_plain.subs(θ2_3, q2 + q3)
    M_plain = M_plain.subs(θ1, q1).subs(θ2, q2).subs(θ3, q3)

    # ------------------------------------------------------------------
    # Build the function string using pycode
    # ------------------------------------------------------------------
    from sympy.printing.pycode import pycode

    params = "q1, q2, q3, m1, m2, m3, l1, l2, l3, r1, r2, r3"
    lines = [
        "import numpy as np",
        "",
        "",
        f"def mass_matrix({params}):",
        '    """',
        "    3x3 mass matrix M(q) for a 3-link planar finger.",
        "    All links are modelled as solid cylinders.",
        "",
        "    Parameters",
        "    ----------",
        "    q1, q2, q3   : joint angles [rad]",
        "    m1, m2, m3   : link masses  [kg]",
        "    l1, l2, l3   : link lengths [m]",
        "    r1, r2, r3   : link radii   [m]",
        "",
        "    Returns",
        "    -------",
        "    M : np.ndarray, shape (3, 3)",
        '    """',
    ]

    rows = []
    for i in range(3):
        row_entries = []
        for j in range(3):
            expr = M_plain[i, j]
            row_entries.append(pycode(expr))
        rows.append(row_entries)

    lines.append("    M = np.array([")
    for i, row in enumerate(rows):
        sep = "," if i < 2 else ""
        lines.append(f"        [{', '.join(row)}]{sep}")
    lines.append("    ])")
    lines.append("    return M")

    func_str = "\n".join(lines)

    # pycode uses math.cos; replace with np.cos for numpy compatibility
    func_str = func_str.replace("math.cos", "np.cos")
    func_str = func_str.replace("math.sin", "np.sin")

    print("\n" + "=" * 60)
    print(func_str)
    print("=" * 60)

    # Copy to clipboard via clip.exe (works on all Windows versions)
    proc = subprocess.run(
        "clip",
        input=func_str.encode("utf-8"),
        capture_output=True,
        shell=True,
    )
    if proc.returncode == 0:
        print("\n[OK] mass_matrix() copied to clipboard.")
    else:
        print("\n[!] Clipboard copy failed — paste the code above manually.")
        print(proc.stderr.decode())


if __name__ == "__main__":
    main()
