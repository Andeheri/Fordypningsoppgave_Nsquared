import sys; sys.path.insert(0,'adaptive_control'); sys.path.insert(0,'.')
import numpy as np
from numpy import sin, cos, pi
from scipy.integrate import solve_ivp
from dynamics import *

wt_frac1=2/3; wt_frac2=1/3
force_s1=l1*0.5; force_s2=l2*0.5; force_s3=l3*0.5
r_circle1=0.010; r_circle2=0.010; r_circle3=0.010
aim_frac1=0.5; aim_frac2=0.5; aim_frac3=0.5
_force_s=(force_s1,force_s2,force_s3); _force_r=(r_circle1,r_circle2,r_circle3)
_force_aim=(aim_frac1,aim_frac2,aim_frac3)

print("At equilibrium, test tau_ext with F=1N:")
s0=[phi1_eq, phi2_eq, phi3_eq, 0.,0.,0.]
tau = tau_link_forces(phi1_eq, phi2_eq, phi3_eq, 0, s0, _force_s, _force_r,
                      np.array([wt_frac1*1.0, wt_frac2*1.0, 0.]), _force_aim)
tau_spring = Tau_K(phi1_eq, phi2_eq, phi3_eq)
print(f"  tau_ext (F=1N) = {tau}")
print(f"  tau_spring at eq = {tau_spring}  (should be ~0)")
print(f"  Signs: positive tau_ext means force OPENS finger further")
print(f"  Negative tau_ext means force CLOSES finger (reduces phi)")

print()
print("Test with F=1N, starting from just above eq:")
def finger_dyn(t, state, F):
    th1,th2,th3,th1d,th2d,th3d = state
    q_dot = np.array([th1d,th2d,th3d])
    M_mat=M(th1,th2,th3); C_mat=C(th1,th2,th3,th1d,th2d,th3d)
    tau_k=Tau_K(th1,th2,th3); tau_b=Tau_B(th1d,th2d,th3d)
    _force_F=np.array((wt_frac1*F, wt_frac2*F, 0.))
    tau_ext=tau_link_forces(th1,th2,th3,t,state,_force_s,_force_r,_force_F,_force_aim)
    rhs=-C_mat@q_dot-tau_k-tau_b+tau_ext
    q_ddot=np.linalg.solve(M_mat,rhs)
    return [th1d,th2d,th3d,q_ddot[0],q_ddot[1],q_ddot[2]]

for F in [0.0, 0.1, 1.0, 5.0]:
    sol=solve_ivp(lambda t,s: finger_dyn(t,s,F),[0.,2.],s0,
                  t_eval=np.linspace(0,2,200),method='RK45',rtol=1e-8,atol=1e-10)
    if sol.success:
        print(f"  F={F}N -> phi1: {np.degrees(phi1_eq):.1f} -> {np.degrees(sol.y[0,-1]):.1f} deg  "
              f"phi2: {np.degrees(phi2_eq):.1f} -> {np.degrees(sol.y[1,-1]):.1f} deg  (success)")
    else:
        print(f"  F={F}N -> FAILED at t={sol.t[-1]:.3f}, phi=[{np.degrees(sol.y[0,-1]):.1f},{np.degrees(sol.y[1,-1]):.1f}] deg")
