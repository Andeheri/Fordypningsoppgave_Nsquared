"""Quick test: verify the Baumgarte-corrected simulation completes and reduces drift."""
import sys; sys.path.insert(0,'adaptive_control'); sys.path.insert(0,'.')
import numpy as np
from numpy import sin, cos, pi
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
from dynamics import *

# ---- Motor parameters ----
Ia_stall=4.9; tau_stall=220*9.81/1000; Ia_no_load=0.2; theta_dot_no_load=130*2*pi/60
Kt=tau_stall/Ia_stall; Bm=Kt*Ia_no_load/theta_dot_no_load; Jm=0.093; r_spindle=0.01

# ---- MRAC ----
omega_n=8.; zeta=1.
A_m=np.array([[0.,1.],[-omega_n**2,-2.*zeta*omega_n]]); B_m=np.array([[0.],[omega_n**2]])
Q=np.diag([10.,1.]); P=solve_continuous_lyapunov(A_m.T,-Q)
Gamma_K=np.diag([20.,3.]); gamma_L=20.; K_0=[0.,0.]; L_0=1.
Ia_stall_clamp=Ia_stall

# ---- Geometry ----
wt_frac1=2/3; wt_frac2=1/3
force_s1=l1*0.5; force_s2=l2*0.5
r_circle1=0.010; r_circle2=0.010
aim_frac1=0.5; aim_frac2=0.5
_force_aim=(aim_frac1,aim_frac2,0.5)

# ---- Baumgarte ----
alpha_B=50.; beta_B=50.
phi1_0=0.; phi2_0=0.
def _cable_len(p1, p2):
    a1=p1; a2=p1+p2
    att1=np.array([force_s1*cos(a1)-r_circle1*sin(a1), force_s1*sin(a1)+r_circle1*cos(a1)])
    att2=np.array([l1*cos(a1)+force_s2*cos(a2)-r_circle2*sin(a2), l1*sin(a1)+force_s2*sin(a2)+r_circle2*cos(a2)])
    t1=np.array([-aim_frac1*l0, 0.]); t2=aim_frac2*l1*np.array([cos(a1),sin(a1)])
    return wt_frac1*np.linalg.norm(t1-att1) + wt_frac2*np.linalg.norm(t2-att2)
C_cable = r_spindle*0. + _cable_len(phi1_0, phi2_0)
print(f"C_cable = {C_cable:.6f} m")

r_ref=lambda t: pi/4*(sin(2*pi*0.3*t)>0).astype(float)

def closed_loop_dynamics(t, z):
    theta,omega=z[0],z[1]; xm=z[2:4]; K=z[4:6]; L=z[6]
    p1,p2,p3=z[7:10]; d1v,d2v,d3v=z[10:13]
    x=np.array([theta,omega]); r_t=r_ref(t); e=x-xm
    i_cmd=float(np.clip(-K@x+L*r_t,-Ia_stall_clamp,Ia_stall_clamp))
    a1=p1; a2=p1+p2
    att1=np.array([force_s1*cos(a1)-r_circle1*sin(a1), force_s1*sin(a1)+r_circle1*cos(a1)])
    att2=np.array([l1*cos(a1)+force_s2*cos(a2)-r_circle2*sin(a2), l1*sin(a1)+force_s2*sin(a2)+r_circle2*cos(a2)])
    t1=np.array([-aim_frac1*l0,0.]); t2=aim_frac2*l1*np.array([cos(a1),sin(a1)])
    def _u(v): n=np.linalg.norm(v); return v/n if n>1e-12 else np.zeros(2)
    dd1=_u(t1-att1); dd2=_u(t2-att2)
    J1=np.array([[-(force_s1*sin(a1)+r_circle1*cos(a1)),0.,0.],[force_s1*cos(a1)-r_circle1*sin(a1),0.,0.]])
    J2=np.array([[-(l1*sin(a1)+force_s2*sin(a2)+r_circle2*cos(a2)),-(force_s2*sin(a2)+r_circle2*cos(a2)),0.],[l1*cos(a1)+force_s2*cos(a2)-r_circle2*sin(a2),force_s2*cos(a2)-r_circle2*sin(a2),0.]])
    Jc=wt_frac1*(dd1@J1)+wt_frac2*(dd2@J2)
    Mm=M(p1,p2,p3); Cm=C(p1,p2,p3,d1v,d2v,d3v); tk=Tau_K(p1,p2,p3); tb=Tau_B(d1v,d2v,d3v)
    qd=np.array([d1v,d2v,d3v]); tp=-Cm@qd-tk-tb
    Mi=np.linalg.inv(Mm)
    H=r_spindle**2/Jm+float(Jc@Mi@Jc)
    Lf=wt_frac1*np.linalg.norm(t1-att1)+wt_frac2*np.linalg.norm(t2-att2)
    gp=r_spindle*theta+Lf-C_cable; gv=r_spindle*omega-float(Jc@qd)
    T=(r_spindle*(Kt*i_cmd-Bm*omega)/Jm - float(Jc@Mi@tp) + 2.*alpha_B*gv + beta_B**2*gp)/H
    qdd=Mi@(tp+Jc*T)
    dth=omega; dom=(Kt*i_cmd-Bm*omega-r_spindle*T)/Jm
    dxm=A_m@xm+B_m.flatten()*r_t
    sig=(B_m.T@P@e.reshape(-1,1)).item()
    dK=Gamma_K@x*sig; dL=-gamma_L*r_t*sig
    return np.hstack(([dth,dom],dxm,dK,[dL],qd,qdd))

z0=[0.,0.,0.,0.,0.,0.,1.,phi1_0,phi2_0,0.,0.,0.,0.]
t_eval=np.linspace(0,10,5000)
sol=solve_ivp(closed_loop_dynamics,[0.,10.],z0,t_eval=t_eval,method='RK45',rtol=1e-6,atol=1e-8)
print(f"success: {sol.success}  steps: {sol.y.shape[1]}")
print(f"theta range: [{sol.y[0].min():.3f}, {sol.y[0].max():.3f}] rad")
print(f"phi1 range (deg): [{np.degrees(sol.y[7].min()):.1f}, {np.degrees(sol.y[7].max()):.1f}]")

# Measure constraint drift over time
Lf_all = np.array([_cable_len(sol.y[7,i], sol.y[8,i]) for i in range(0, len(sol.t), 50)])
g_pos_all = r_spindle*sol.y[0,::50] + Lf_all - C_cable
print(f"\nConstraint drift g_pos: max={np.abs(g_pos_all).max():.2e} m, mean={np.abs(g_pos_all).mean():.2e} m")
print(f"(smaller = better; 1e-6 m is excellent, 1e-3 m is acceptable)")
