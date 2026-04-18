
import sympy as sp

J, c, k, Ra, La, Kb, Kt = sp.symbols('J c k Ra La Kb Kt')
s = sp.Symbol('s')
theta = sp.Function('theta')(s)
Ea = sp.Function('Ea')(s)



lhs = (J*s**2 + c*s + k) * theta

a = lhs * (Ra+La*s)

sp.pprint(sp.collect(a.expand(), s))