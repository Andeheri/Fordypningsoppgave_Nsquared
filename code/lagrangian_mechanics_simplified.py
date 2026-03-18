
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

class ParameterSubstituter:
    """
    Replaces angle-independent (constant) coefficient sub-expressions in
    SymPy matrices/expressions with compact p_i symbols.

    The mapping is preserved across calls, so calling substitute() on M and
    then on C produces consistent p-numbering (same physical constant → same p).

    Usage
    -----
    ps = ParameterSubstituter([θ1, θ2, θ3])
    M_p = ps.substitute(M)
    C_p = ps.substitute(C)
    ps.print_definitions()
    """

    def __init__(self, angle_syms: list):
        """
        :param angle_syms: list of angle symbols treated as "variable" (everything
                           else is considered a constant coefficient).
        """
        from collections import defaultdict as _dd
        self._dd = _dd
        self.angle_syms = angle_syms
        self._const_exprs: list = []        # in creation order
        self._seen: dict        = {}        # str(expr) -> p_symbol

    # ------------------------------------------------------------------
    def _get_p(self, c: sp.Expr) -> sp.Expr:
        """Return existing p-symbol (or -p) for c, creating one if needed."""
        c = sp.simplify(c)
        if c == sp.Integer(0):
            return sp.Integer(0)
        key     = str(c)
        neg_key = str(sp.simplify(-c))
        if key in self._seen:
            return self._seen[key]
        if neg_key in self._seen:
            return -self._seen[neg_key]
        idx = len(self._const_exprs) + 1
        p = sp.Symbol(f'p{idx}', positive=True, real=True)
        self._const_exprs.append(c)
        self._seen[key] = p
        return p

    # ------------------------------------------------------------------
    def substitute(self, expr) -> sp.Expr:
        """
        Replace constant coefficients in *expr* (scalar or Matrix) with p-symbols.
        Returns the compactified expression/matrix.
        """
        if isinstance(expr, sp.MatrixBase):
            return sp.Matrix(
                expr.rows, expr.cols,
                lambda i, j: self.substitute(expr[i, j])
            )
        # Decompose scalar into trig-basis terms, group by basis
        d = self._dd(lambda: sp.Integer(0))
        for term in sp.Add.make_args(sp.expand(expr)):
            coeff, trig_part = term.as_independent(*self.angle_syms, as_Add=False)
            d[trig_part] += coeff
        result = sp.Integer(0)
        for trig_part, coeff in d.items():
            result += self._get_p(sp.simplify(coeff)) * trig_part
        return result

    # ------------------------------------------------------------------
    def print_definitions(self):
        """Pretty-print all p_i = <constant expression> definitions."""
        print("\nParameter definitions:")
        for expr in self._const_exprs:
            p = self._seen[str(expr)]
            pretty_str = sp.pretty(expr, use_unicode=True)
            lines = pretty_str.split('\n')
            baseline = len(lines) // 2
            prefix = f"  {p} = "
            pad    = " " * len(prefix)
            for k, line in enumerate(lines):
                print((prefix if k == baseline else pad) + line)


# Display substitutions: Derivative objects → compact Unicode dot-notation symbols
_DOT_DISPLAY_SUBS = {
    θ1_dot:   sp.Symbol('θ\u0307\u2081'),      # θ̇₁
    θ2_dot:   sp.Symbol('θ\u0307\u2082'),      # θ̇₂
    θ3_dot:   sp.Symbol('θ\u0307\u2083'),      # θ̇₃
    θ1_2_dot: sp.Symbol('θ\u0307\u2081\u2082'),  # θ̇₁₂
    θ2_3_dot: sp.Symbol('θ\u0307\u2082\u2083'),  # θ̇₂₃
    θ1_3_dot: sp.Symbol('θ\u0307\u2081\u2083'),  # θ̇₁₃
    θ1:       sp.Symbol('θ\u2081'),            # θ₁
    θ2:       sp.Symbol('θ\u2082'),            # θ₂
    θ3:       sp.Symbol('θ\u2083'),            # θ₃
    θ1_2:     sp.Symbol('θ\u2081\u2082'),      # θ₁₂
    θ2_3:     sp.Symbol('θ\u2082\u2083'),      # θ₂₃
    θ1_3:     sp.Symbol('θ\u2081\u2083'),      # θ₁₃
}


def dot_pprint(expr, **kwargs):
    """
    Pretty-print *expr* using dot notation for time derivatives.
    Derivative(theta_i(t), t) is rendered as θ̇ᵢ instead of the default
    d(theta_i(t))/dt form.
    """
    kwargs.setdefault('use_unicode', True)
    if isinstance(expr, sp.MatrixBase):
        display_expr = expr.applyfunc(lambda e: e.subs(_DOT_DISPLAY_SUBS))
    else:
        display_expr = expr.subs(_DOT_DISPLAY_SUBS)
    sp.pprint(display_expr, **kwargs)


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

    # J1 = inertia_cylinder(m1, r1, l1)
    # J2 = inertia_cylinder(m2, r2, l2)
    # J3 = inertia_cylinder(m3, r3, l3)
    J1, J2, J3 = sp.symbols('J1 J2 J3', positive=True, real=True)

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
    T = sp.Rational(1, 2) * (J1 * θ1_dot**2 + J2 * (θ1_dot + θ2_dot)**2 + J3 * (θ1_dot + θ2_dot + θ3_dot)**2 + (r_CM_dot_vec.T * M6 * r_CM_dot_vec)[0, 0])
    # Unpack kinetic energy
    T = sp.simplify(T)

    print("\nKinetic energy T:")
    dot_pprint(T)
    
    progress_bar.update(1)
    progress_bar.set_description("Computing the generalized inertia matrix")

    M = sp.hessian(T, theta_dot_vec)   
    M = sp.simplify(M)

    # M = substitute_angles(M)
    progress_bar.update(1)
    print("\nMass matrix M(q):")
    dot_pprint(M)
    dot_pprint(M[0, 0])

    # --- Substitute constant sub-expressions with p1, p2, ... ---
    ps = ParameterSubstituter([θ1, θ2, θ3])
    M = ps.substitute(M)
    ps.print_definitions()
    print("\nMass matrix M(q) with p-parameters:")
    dot_pprint(M)

    progress_bar.set_description("Computing Coriolis forces")
    # Coriolis forces
    C = sp.Matrix(3, 3, lambda i, j: sum(
        sp.Rational(1, 2) * (sp.diff(M[i, j], theta_vec[k]) + sp.diff(M[i, k], theta_vec[j]) - sp.diff(M[j, k], theta_vec[i])) * theta_dot_vec[k]
        for k in range(3)
    ))
    C = sp.simplify(C)
    progress_bar.update(1)
    print("\nCoriolis matrix C(q, q_dot):")
    dot_pprint(C)

    progress_bar.close()
    return

    # ---- Export M and C as callable Python functions to a text file ----
    print("\nExporting M and C as Python functions to 'dynamics_functions.py'...")

    # Plain (non-time-dependent) symbols used in the output
    th1, th2, th3 = sp.symbols('theta1 theta2 theta3', real=True)
    th1d, th2d, th3d = sp.symbols('theta1_dot theta2_dot theta3_dot', real=True)

    # Step 1: expand combined-angle shorthands back to their components
    combined_angle_subs = {
        θ1_3: θ1 + θ2 + θ3,
        θ1_2: θ1 + θ2,
        θ2_3: θ2 + θ3,
    }
    combined_dot_subs = {
        θ1_3_dot: θ1_dot + θ2_dot + θ3_dot,
        θ1_2_dot: θ1_dot + θ2_dot,
        θ2_3_dot: θ2_dot + θ3_dot,
    }

    M_out = M.subs(combined_angle_subs)
    C_out = C.subs({**combined_angle_subs, **combined_dot_subs})

    # Step 2: replace time-dependent sympy functions with plain symbols
    func_subs = {θ1: th1, θ2: th2, θ3: th3,
                 θ1_dot: th1d, θ2_dot: th2d, θ3_dot: th3d}
    M_out = M_out.subs(func_subs)
    C_out = C_out.subs(func_subs)

    output_file = "dynamics_functions_temp.py"
    with open(output_file, "w") as f:
        f.write("from numpy import sin, cos, array\n\n\n")

        # M(theta1, theta2, theta3)
        f.write("def M(theta1, theta2, theta3):\n")
        f.write("    return array([\n")
        for i in range(3):
            row = [str(M_out[i, j]) for j in range(3)]
            f.write(f"        [{', '.join(row)}],\n")
        f.write("    ])\n\n\n")

        # C(theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot)
        f.write("def C(theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot):\n")
        f.write("    return array([\n")
        for i in range(3):
            row = [str(C_out[i, j]) for j in range(3)]
            f.write(f"        [{', '.join(row)}],\n")
        f.write("    ])\n")

    print(f"Saved to '{output_file}'.")

if __name__ == "__main__":
    main()