import sys; sys.path.insert(0,'adaptive_control'); sys.path.insert(0,'.')
import numpy as np
from numpy import sin, cos, pi
from dynamics import *

wt_frac1=2/3; wt_frac2=1/3
force_s1=l1*0.5; force_s2=l2*0.5
r_circle1=0.010; r_circle2=0.010
aim_frac1=0.5; aim_frac2=0.5
r_spindle=0.01

a1=phi1_eq; a2=phi1_eq+phi2_eq
MCP=np.array([l1*cos(a1), l1*sin(a1)])
att1=np.array([force_s1*cos(a1)-r_circle1*sin(a1), force_s1*sin(a1)+r_circle1*cos(a1)])
att2=MCP+np.array([force_s2*cos(a2)-r_circle2*sin(a2), force_s2*sin(a2)+r_circle2*cos(a2)])
target1=np.array([-aim_frac1*l0, 0.]); target2=aim_frac2*l1*np.array([cos(a1),sin(a1)])
def _u(v): n=np.linalg.norm(v); return v/n if n>1e-12 else np.zeros(2)
d1=_u(target1-att1); d2=_u(target2-att2)
J1=np.array([[-(force_s1*sin(a1)+r_circle1*cos(a1)),0.,0.],[force_s1*cos(a1)-r_circle1*sin(a1),0.,0.]])
J2=np.array([[-(l1*sin(a1)+force_s2*sin(a2)+r_circle2*cos(a2)),-(force_s2*sin(a2)+r_circle2*cos(a2)),0.],[l1*cos(a1)+force_s2*cos(a2)-r_circle2*sin(a2),force_s2*cos(a2)-r_circle2*sin(a2),0.]])
J_cable = wt_frac1*(d1@J1) + wt_frac2*(d2@J2)
print('J_cable at equilibrium:', J_cable)
print('r_spindle:', r_spindle)
print()
print('Quasi-static: r_spindle*dtheta = J_cable @ dphi')
print('  d(phi1)/d(theta) =', r_spindle/J_cable[0], 'rad/rad')
print('  d(phi2)/d(theta) =', r_spindle/J_cable[1], 'rad/rad')
print()
print(f'  For theta 0->pi/4 ({np.degrees(pi/4):.0f} deg):')
print(f'    phi1 ~+{np.degrees(pi/4*r_spindle/J_cable[0]):.1f} deg')
print(f'    phi2 ~+{np.degrees(pi/4*r_spindle/J_cable[1]):.1f} deg')
print()
g=9.81; Ia_stall=4.9; tau_stall=220*g/1000; theta_dot_no_load=130*2*pi/60; Ia_no_load=0.2
Kt=tau_stall/Ia_stall
F_1A = Kt*1.0/r_spindle
tau_ext_1A = J_cable * F_1A
print(f'Cable torque at 1A (no constraint): {tau_ext_1A} N*m')
print(f'For 10 deg deflection at 1A stall:')
print(f'  k1 required = {tau_ext_1A[0]/(10*pi/180):.4f} N*m/rad')
print(f'  k2 required = {tau_ext_1A[1]/(10*pi/180):.4f} N*m/rad')
print(f'  current k = 0.05 N*m/rad')
