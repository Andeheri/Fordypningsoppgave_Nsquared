import sys; sys.path.insert(0,'adaptive_control'); sys.path.insert(0,'.')
import numpy as np
from numpy import sin, cos, pi
from dynamics import *

wt_frac1=2/3; wt_frac2=1/3
force_s1=l1*0.5; force_s2=l2*0.5; force_s3=l3*0.5
r_circle1=0.010; r_circle2=0.010; r_circle3=0.010
aim_frac1=0.5; aim_frac2=0.5; aim_frac3=0.5
_force_s=(force_s1,force_s2,force_s3); _force_r=(r_circle1,r_circle2,r_circle3)
_force_aim=(aim_frac1,aim_frac2,aim_frac3)

# Compute the whiffle tree force from virtual work / cable_tensions.
# Alternative approach: use tau_link_forces Jacobian directly.
# For the whiffle tree, motor cable force = sum of component forces delivered to links 1 and 2.
# By virtual work: tau_motor = r_spindle * F_motor
# F_motor acts on the whiffle tree in a direction that splits as wt_frac1 to link1, wt_frac2 to link2.
# The reaction back on the motor is F_reaction = wt_frac1*T1 + wt_frac2*T2
# where T_i is the tension each sub-cable exerts.
# But cable_tensions solves for T from passive torques using the geometry matrix G.

# Let's check: what does cable_tensions return vs spring torques over phi1 range?
print("phi1 (deg)  phi2 (deg)  T1(N)   T2(N)   F_motor(N)  spring_tau1  cond(G)")
for phi1 in np.linspace(0, pi/2, 10):
    for phi2 in np.linspace(0, pi/2, 10):
        phi3 = phi3_eq
        state = [phi1, phi2, phi3, 0, 0, 0]
        T = cable_tensions(state, _force_s, _force_r, _force_aim)
        F_mot = wt_frac1*T[0] + wt_frac2*T[1]
        tau_p = Tau_K(phi1,phi2,phi3)
        # Check cond(G)
        a1=phi1; a2=phi1+phi2; a3=phi1+phi2+phi3
        MCP=np.array([l1*cos(a1), l1*sin(a1)])
        PIP=MCP+np.array([l2*cos(a2), l2*sin(a2)])
        att1=np.array([force_s1*cos(a1)-r_circle1*sin(a1), force_s1*sin(a1)+r_circle1*cos(a1)])
        att2=MCP+np.array([force_s2*cos(a2)-r_circle2*sin(a2), force_s2*sin(a2)+r_circle2*cos(a2)])
        att3=PIP+np.array([force_s3*cos(a3)-r_circle3*sin(a3), force_s3*sin(a3)+r_circle3*cos(a3)])
        target1=np.array([-aim_frac1*l0,0.]); target2=aim_frac2*l1*np.array([cos(a1),sin(a1)]); target3=MCP+aim_frac3*l2*np.array([cos(a2),sin(a2)])
        def _u(v): n=np.linalg.norm(v); return v/n if n>1e-12 else np.zeros(2)
        d1=_u(target1-att1); d2=_u(target2-att2); d3=_u(target3-att3)
        J1=np.array([[-(force_s1*sin(a1)+r_circle1*cos(a1)),0,0],[force_s1*cos(a1)-r_circle1*sin(a1),0,0]])
        J2=np.array([[-(l1*sin(a1)+force_s2*sin(a2)+r_circle2*cos(a2)),-(force_s2*sin(a2)+r_circle2*cos(a2)),0],[l1*cos(a1)+force_s2*cos(a2)-r_circle2*sin(a2),force_s2*cos(a2)-r_circle2*sin(a2),0]])
        J3=np.array([[-(l1*sin(a1)+l2*sin(a2)+force_s3*sin(a3)+r_circle3*cos(a3)),-(l2*sin(a2)+force_s3*sin(a3)+r_circle3*cos(a3)),-(force_s3*sin(a3)+r_circle3*cos(a3))],[l1*cos(a1)+l2*cos(a2)+force_s3*cos(a3)-r_circle3*sin(a3),l2*cos(a2)+force_s3*cos(a3)-r_circle3*sin(a3),force_s3*cos(a3)-r_circle3*sin(a3)]])
        G=np.column_stack([J1.T@d1, J2.T@d2, J3.T@d3])
        cond=np.linalg.cond(G)
        if abs(F_mot) > 50 or cond > 1e6:
            print(f"  phi1={np.degrees(phi1):.0f} phi2={np.degrees(phi2):.0f}  T=[{T[0]:.2f},{T[1]:.2f},{T[2]:.2f}]  F_mot={F_mot:.2f}  tau_k={tau_p}  cond={cond:.1e}  *** LARGE ***")

print("done - no output above means all configs are well-conditioned with reasonable forces")

# Now check: is there a simpler formula?
# The total torque from the whiffle-tree cable on joint 1 is
# tau_wt = r_spindle * F_motor * (effective moment arm)
# But actually: the force from the motor is what drives tau_ext in the finger.
# The REACTION on the motor is the same force by Newton's 3rd law.
# The total force vector along the cable = output_force
# The torque on the motor = r_spindle * output_force (this is already in B_true*i_cmd!)
# The ADDITIONAL load torque is from the PASSIVE spring/damper restoring torques.
# This = r_spindle * F_passive_reaction
# where F_passive_reaction is what the springs "push back" through the cables.
# That is exactly cable_tensions.

# Let's check how cable_tensions behaves at the equilibrium states of the finger
print("\nAt equilibrium phi=phi_eq:")
T_eq = cable_tensions([phi1_eq, phi2_eq, phi3_eq, 0,0,0], _force_s, _force_r, _force_aim)
print(f"  T_eq = {T_eq}")
print(f"  F_motor_eq = {wt_frac1*T_eq[0]+wt_frac2*T_eq[1]:.4f} N")
print(f"  (Should be ~0 since springs are at rest)")
