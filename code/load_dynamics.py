import dill
import sympy as sp

with open('dynamics_results.pkl', 'rb') as f:
    data = dill.load(f)

M = data['M']
C = data['C']
p_defs = data['p_defs']

# Re-declare each p as a positive symbol so SymPy can use the assumption
positive_subs = {p: sp.Symbol(p.name, positive=True, real=True) for p, _ in p_defs}
M = M.subs(positive_subs)
C = C.subs(positive_subs)
p_defs = [(positive_subs[p], expr) for p, expr in p_defs]

print("M matrix:")
sp.pprint(M)

print("\nC matrix:")
sp.pprint(C)

print("\np definitions:")
for p, expr in p_defs:
    sp.pprint(sp.Eq(p, expr))

# print("\nInverse of M:")
# M_inv = sp.simplify(M.inv())
# sp.pprint(M_inv)
