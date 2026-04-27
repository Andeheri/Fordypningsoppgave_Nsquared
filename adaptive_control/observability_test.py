
import sympy as sp

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

C = sp.Matrix([
    [1, 0, 0],
    [0, 0, 1]
])

observability_matrix = sp.Matrix.vstack(C, C @ A, C @ A @ A)
observability_matrix = observability_matrix.applyfunc(lambda x: sp.simplify(x))
rank_of_observability_matrix = observability_matrix.rank()
print("Observability matrix:")
sp.pprint(observability_matrix, use_unicode=True)
print(f"Rank of observability matrix: {rank_of_observability_matrix}")