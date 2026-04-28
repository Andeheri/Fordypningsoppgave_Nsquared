import sys; sys.path.insert(0,'adaptive_control'); sys.path.insert(0,'.')
import numpy as np
from numpy import sin, cos, pi
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
from dynamics import *

V_supply=12.0; g=9.81
Ia_stall=4.9; tau_stall=220*g/1000
Ia_no_load=0.2; theta_dot_no_load=130*2*pi/60
Kt=tau_stall/Ia_stall; Ra=V_supply/Ia_stall
Bm=Kt*Ia_no_load/theta_dot_no_load
Jm=0.093; r_spindle=0.01
omega_n=8.; zeta=1.
A_m=np.array([[0.,1.],[-omega_n**2,-2.*zeta*omega_n]]); B_m=np.array([[0.],[omega_n**2]])
Q=np.diag([10.,1.]); P=solve_continuous_lyapunov(A_m.T,-Q)
Gamma_K=np.diag([20.,3.]); gamma_L=20.
K_0=[0.,0.]; L_0=1.
wt_frac1=2/3; wt_frac2=1/3
force_s1=l1*0.5; force_s2=l2*0.5
r_circle1=0.010; r_circle2=0.010
aim_frac1=0.5; aim_frac2=0.5
_force_aim=(aim_frac1,aim_frac2,0.5)
Ia_stall_clamp = Ia_stall
r=lambda t: pi/4*(sin(2*pi*0.3*t)>0).astype(float)

def closed_loop_dynamics(t,z):
    theta,omega=z[0],z[1]; xm=z[2:4]; K=z[4:6]; L=z[6]
    phi_1,phi_2,phi_3=z[7:10]; phi_1_dot,phi_2_dot,phi_3_dot=z[10:13]
    x=np.array([theta,omega]); r_t=r(t); e=x-xm
    i_cmd=float(np.clip(-K@x+L*r_t,-Ia_stall_clamp,Ia_stall_clamp))

    a1=phi_1; a2=phi_1+phi_2
    MCP=np.array([l1*cos(a1), l1*sin(a1)])
    att1=np.array([force_s1*cos(a1)-r_circle1*sin(a1), force_s1*sin(a1)+r_circle1*cos(a1)])
    att2=MCP+np.array([force_s2*cos(a2)-r_circle2*sin(a2), force_s2*sin(a2)+r_circle2*cos(a2)])
    target1=np.array([-_force_aim[0]*l0, 0.]); target2=_force_aim[1]*l1*np.array([cos(a1),sin(a1)])
    def _u(v): n=np.linalg.norm(v); return v/n if n>1e-12 else np.zeros(2)
    d1=_u(target1-att1); d2=_u(target2-att2)
    J1=np.array([[-(force_s1*sin(a1)+r_circle1*cos(a1)),0.,0.],[force_s1*cos(a1)-r_circle1*sin(a1),0.,0.]])
    J2=np.array([[-(l1*sin(a1)+force_s2*sin(a2)+r_circle2*cos(a2)),-(force_s2*sin(a2)+r_circle2*cos(a2)),0.],[l1*cos(a1)+force_s2*cos(a2)-r_circle2*sin(a2),force_s2*cos(a2)-r_circle2*sin(a2),0.]])
    J_cable=wt_frac1*(d1@J1)+wt_frac2*(d2@J2)

    M_mat=M(phi_1,phi_2,phi_3); C_mat=C(phi_1,phi_2,phi_3,phi_1_dot,phi_2_dot,phi_3_dot)
    tau_k=Tau_K(phi_1,phi_2,phi_3); tau_b=Tau_B(phi_1_dot,phi_2_dot,phi_3_dot)
    q_dot=np.array([phi_1_dot,phi_2_dot,phi_3_dot])
    tau_passive=-C_mat@q_dot-tau_k-tau_b

    M_inv=np.linalg.inv(M_mat)
    H=r_spindle**2/Jm+float(J_cable@M_inv@J_cable)
    T_cable=(r_spindle*(Kt*i_cmd-Bm*omega)/Jm - float(J_cable@M_inv@tau_passive))/H

    q_ddot=M_inv@(tau_passive+J_cable*T_cable)
    dtheta=omega; domega=(Kt*i_cmd-Bm*omega-r_spindle*T_cable)/Jm
    dxm=A_m@xm+B_m.flatten()*r_t
    sigma=(B_m.T@P@e.reshape(-1,1)).item()
    dK=Gamma_K@x*sigma; dL=-gamma_L*r_t*sigma
    return np.hstack(([dtheta,domega],dxm,dK,[dL],q_dot,q_ddot))

z0=[0.,0.,0.,0.,K_0[0],K_0[1],L_0,phi1_eq,phi2_eq,phi3_eq,0.,0.,0.]
sol=solve_ivp(closed_loop_dynamics,[0.,10.],z0,t_eval=np.linspace(0,10,5000),method='RK45',rtol=1e-6,atol=1e-8)
print('success:', sol.success)
print('message:', sol.message)
print('steps:', sol.y.shape[1])
t=sol.t; phi=sol.y[7:10]; theta=sol.y[0]
print(f'theta range: [{theta.min():.3f}, {theta.max():.3f}] rad')
print(f'phi1 range (deg): [{np.degrees(phi[0].min()):.1f}, {np.degrees(phi[0].max()):.1f}]')
print(f'phi2 range (deg): [{np.degrees(phi[1].min()):.1f}, {np.degrees(phi[1].max()):.1f}]')
print(f'Check proportionality - corr(theta, phi1): {np.corrcoef(theta, phi[0])[0,1]:.4f}')
print(f'Check proportionality - corr(theta, phi2): {np.corrcoef(theta, phi[1])[0,1]:.4f}')
