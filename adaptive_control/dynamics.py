
import numpy as np
from numpy import sin, cos, pi

m1, m2, m3 = 0.02, 0.015, 0.020  # Masses [kg]
l1, l2, l3 = 0.048, 0.030, 0.024  # Lengths [m]
l0 = 0.048  # Metacarpal length [m]
r1, r2, r3 = 0.085, 0.085, 0.085  # Radii [m]
N_max = 22 # Maybe 30
cable_speed = 67  # mm/s
k1, k2, k3 = 2, 2, 2  # Spring constants [N*m/rad]
b1, b2, b3 = 0.05, 0.05, 0.05  # Damping coefficients [N*m*s/rad]

phi1_eq, phi2_eq, phi3_eq = pi/6, pi/4, pi/12  # Spring rest angles [rad]

def Tau_K(phi1, phi2, phi3):
    """
    3x1 vector of spring forces.
    
    Parameters
    ----------
    phi1, phi2, phi3   : Relative joint angles [rad]
    phi1_eq, phi2_eq, phi3_eq : spring rest angles [rad]
    k1, k2, k3   : spring constants [N*m/rad]

    Returns
    -------
    Tau_K : np.ndarray, shape (3, 1)
    """
    Tau_K = np.array([
        k1*(phi1 - phi1_eq),
        k2*(phi2 - phi2_eq),
        k3*(phi3 - phi3_eq),
    ])
    return Tau_K


def Tau_B(phi1_dot, phi2_dot, phi3_dot):
    """
    3x1 vector of damping forces.
    
    Parameters
    ----------
    phi1_dot, phi2_dot, phi3_dot : joint angular velocities [rad/s]
    b1, b2, b3   : damping coefficients [N*m*s/rad]

    Returns
    -------
    Tau_B : np.ndarray, shape (3, 1)
    """
    Tau_B = np.array([
        b1*phi1_dot,
        b2*phi2_dot,
        b3*phi3_dot,
    ])
    return Tau_B


def C(phi1, phi2, phi3, phi1_dot, phi2_dot, phi3_dot):
    """
    3x3 Coriolis and centrifugal matrix C(q, q_dot) for a 3-link planar finger.
    All links are modelled as solid cylinders.
    
    Parameters
    ----------
    phi1, phi2, phi3   : joint angles [rad]
    phi1_dot, phi2_dot, phi3_dot : joint angular velocities [rad/s]
    m1, m2, m3   : link masses  [kg]
    l1, l2, l3   : link lengths [m]
    r1, r2, r3   : link radii   [m]
    
    Returns
    -------
    C : np.ndarray, shape (3, 3)
    """
    return np.array([
        [-l1*phi2_dot*(l2*m2*sin(phi2) + 2*l2*m3*sin(phi2) + l3*m3*sin(phi2 + phi3))/2 - l3*m3*phi3_dot*(l1*sin(phi2 + phi3) + l2*sin(phi3))/2, -l1*phi1_dot*(l2*m2*sin(phi2) + 2*l2*m3*sin(phi2) + l3*m3*sin(phi2 + phi3))/2 - l1*phi2_dot*(l2*m2*sin(phi2) + 2*l2*m3*sin(phi2) + l3*m3*sin(phi2 + phi3))/2 - l3*m3*phi3_dot*(l1*sin(phi2 + phi3) + l2*sin(phi3))/2, l3*m3*(l1*sin(phi2 + phi3) + l2*sin(phi3))*(-phi1_dot - phi2_dot - phi3_dot)/2],
        [l1*phi1_dot*(l2*m2*sin(phi2) + 2*l2*m3*sin(phi2) + l3*m3*sin(phi2 + phi3))/2 - l2*l3*m3*phi3_dot*sin(phi3)/2, -l2*l3*m3*phi3_dot*sin(phi3)/2, l2*l3*m3*(-phi1_dot - phi2_dot - phi3_dot)*sin(phi3)/2],
        [l3*m3*(l2*phi2_dot*sin(phi3) + phi1_dot*(l1*sin(phi2 + phi3) + l2*sin(phi3)))/2, l2*l3*m3*(phi1_dot + phi2_dot)*sin(phi3)/2, 0],
    ])


def M(phi1, phi2, phi3):
    """
    3x3 mass matrix M(q) for a 3-link planar finger.
    All links are modelled as solid cylinders.

    Parameters
    ----------
    phi1, phi2, phi3   : joint angles [rad]
    m1, m2, m3   : link masses  [kg]
    l1, l2, l3   : link lengths [m]
    r1, r2, r3   : link radii   [m]

    Returns
    -------
    M : np.ndarray, shape (3, 3)
    """
    return np.array([
        [7*l1**2*m1/12 + l1**2*m2 + l1**2*m3 + l1*l2*m2*cos(phi2) + 2*l1*l2*m3*cos(phi2) + l1*l3*m3*cos(phi2 + phi3) + 7*l2**2*m2/12 + l2**2*m3 + l2*l3*m3*cos(phi3) + 7*l3**2*m3/12 + m1*r1**2/4 + m2*r2**2/4 + m3*r3**2/4, l1*l2*m2*cos(phi2)/2 + l1*l2*m3*cos(phi2) + l1*l3*m3*cos(phi2 + phi3)/2 + 7*l2**2*m2/12 + l2**2*m3 + l2*l3*m3*cos(phi3) + 7*l3**2*m3/12 + m2*r2**2/4 + m3*r3**2/4, m3*(6*l1*l3*cos(phi2 + phi3) + 6*l2*l3*cos(phi3) + 7*l3**2 + 3*r3**2)/12],
        [l1*l2*m2*cos(phi2)/2 + l1*l2*m3*cos(phi2) + l1*l3*m3*cos(phi2 + phi3)/2 + 7*l2**2*m2/12 + l2**2*m3 + l2*l3*m3*cos(phi3) + 7*l3**2*m3/12 + m2*r2**2/4 + m3*r3**2/4, 7*l2**2*m2/12 + l2**2*m3 + l2*l3*m3*cos(phi3) + 7*l3**2*m3/12 + m2*r2**2/4 + m3*r3**2/4, m3*(6*l2*l3*cos(phi3) + 7*l3**2 + 3*r3**2)/12],
        [m3*(6*l1*l3*cos(phi2 + phi3) + 6*l2*l3*cos(phi3) + 7*l3**2 + 3*r3**2)/12, m3*(6*l2*l3*cos(phi3) + 7*l3**2 + 3*r3**2)/12, m3*(7*l3**2 + 3*r3**2)/12],
    ])


def tau_link_forces(phi1, phi2, phi3, t, state, force_s, r_circle, F_link, aim_frac):
    """
    Generalized torques from three link forces, via the virtual-work principle.

    A separate force acts on each link.  Its attachment point is the link
    midpoint displaced radially outward by the circle radius (90° CCW from
    the link axis).  The force direction is computed automatically so that it
    points from the attachment point towards a user-specified target on the
    link below (or on the metacarpal for link 1):

        target3  →  MCP + aim3 * l2 * [cos(a2), sin(a2)]    (on link 2)
        target2  →  base + aim2 * l1 * [cos(a1), sin(a1)]   (on link 1)
        target1  →  [-aim1 * l0, 0]                          (on metacarpal)

    The world-frame force angle for link i is therefore

        phi_i = atan2(target_i - att_i)

    and the force vector is

        F_i = F_link_i(t) * [cos(phi_i), sin(phi_i)]

    where the cumulative joint angles are

        a1 = phi1,   a2 = phi1+phi2,   a3 = phi1+phi2+phi3.

    Attachment-point world coordinates
    -----------------------------------
    Each attachment point is the midpoint along the link, offset radially outward
    (90° CCW from the link axis) by the circle radius r_circle_i:

        p_att1 = force_s1*[cos(a1), sin(a1)]  +  r_circle1*[-sin(a1), cos(a1)]
        p_att2 = l1*[cos(a1), sin(a1)]  +  force_s2*[cos(a2), sin(a2)]  +  r_circle2*[-sin(a2), cos(a2)]
        p_att3 = l1*[cos(a1), sin(a1)]  +  l2*[cos(a2), sin(a2)]  +  force_s3*[cos(a3), sin(a3)]  +  r_circle3*[-sin(a3), cos(a3)]

    Positional Jacobians  J_i  (2x3)  of each attachment point
    -----------------------------------------------------------
        J1 = [[ -(force_s1*sin(a1) + r_circle1*cos(a1)),                                                                  0,                                             0 ],
              [  force_s1*cos(a1) - r_circle1*sin(a1),                                                                    0,                                             0 ]]

        J2 = [[ -(l1*sin(a1) + force_s2*sin(a2) + r_circle2*cos(a2)),  -(force_s2*sin(a2) + r_circle2*cos(a2)),           0 ],
              [  l1*cos(a1) + force_s2*cos(a2) - r_circle2*sin(a2),     force_s2*cos(a2) - r_circle2*sin(a2),             0 ]]

        J3 = [[ -(l1*sin(a1) + l2*sin(a2) + force_s3*sin(a3) + r_circle3*cos(a3)),  -(l2*sin(a2) + force_s3*sin(a3) + r_circle3*cos(a3)),  -(force_s3*sin(a3) + r_circle3*cos(a3)) ],
              [  l1*cos(a1) + l2*cos(a2) + force_s3*cos(a3) - r_circle3*sin(a3),     l2*cos(a2) + force_s3*cos(a3) - r_circle3*sin(a3),     force_s3*cos(a3) - r_circle3*sin(a3)  ]]

    Generalized torques (virtual-work principle)
    -------------------------------------------
        tau_i     = J_i^T @ F_i
        tau_total = tau_1 + tau_2 + tau_3

    Parameters
    ----------
    phi1, phi2, phi3 : joint angles [rad]
    t                      : current simulation time [s]
    aim_frac               : tuple (aim1, aim2, aim3) - fractional position (0-1)
                             along the link below where each force aims.

    Returns
    -------
    tau : np.ndarray, shape (3,)   generalized torques [N*m]
    """
    force_s1, force_s2, force_s3 = force_s
    r_circle1, r_circle2, r_circle3 = r_circle
    F_link1, F_link2, F_link3 = F_link
    aim1, aim2, aim3 = aim_frac

    a1 = phi1
    a2 = phi1 + phi2
    a3 = phi1 + phi2 + phi3

    # World-frame joint positions
    MCP = np.array([l1*cos(a1), l1*sin(a1)])
    PIP = MCP + np.array([l2*cos(a2), l2*sin(a2)])

    # Attachment points (midpoint + radial circle offset, 90° CCW from link axis)
    att1 = np.array([force_s1*cos(a1) - r_circle1*sin(a1),
                     force_s1*sin(a1) + r_circle1*cos(a1)])
    att2 = MCP + np.array([force_s2*cos(a2) - r_circle2*sin(a2),
                           force_s2*sin(a2) + r_circle2*cos(a2)])
    att3 = PIP + np.array([force_s3*cos(a3) - r_circle3*sin(a3),
                           force_s3*sin(a3) + r_circle3*cos(a3)])

    # Target points on the link below (or metacarpal)
    target1 = np.array([-aim1 * l0, 0.0])                        # on metacarpal
    target2 = aim2 * l1 * np.array([cos(a1), sin(a1)])           # on link 1
    target3 = MCP + aim3 * l2 * np.array([cos(a2), sin(a2)])     # on link 2

    # World-frame force angles (from attachment point towards target)
    phi1 = np.arctan2(target1[1] - att1[1], target1[0] - att1[0])
    phi2 = np.arctan2(target2[1] - att2[1], target2[0] - att2[0])
    phi3 = np.arctan2(target3[1] - att3[1], target3[0] - att3[0])

    # ---- Link 1 force ----
    # Attachment: p_att1 = force_s1*[cos(a1), sin(a1)] + r_circle1*[-sin(a1), cos(a1)]
    J1 = np.array([
        [-(force_s1*sin(a1) + r_circle1*cos(a1)),  0.0,  0.0],
        [ force_s1*cos(a1) - r_circle1*sin(a1),    0.0,  0.0],
    ])
    F1 = F_link1 * np.array([cos(phi1), sin(phi1)])
    tau1 = J1.T @ F1

    # ---- Link 2 force ----
    # Attachment: p_att2 = l1*[cos(a1), sin(a1)] + force_s2*[cos(a2), sin(a2)] + r_circle2*[-sin(a2), cos(a2)]
    J2 = np.array([
        [-(l1*sin(a1) + force_s2*sin(a2) + r_circle2*cos(a2)),  -(force_s2*sin(a2) + r_circle2*cos(a2)),  0.0],
        [ l1*cos(a1) + force_s2*cos(a2) - r_circle2*sin(a2),     force_s2*cos(a2) - r_circle2*sin(a2),   0.0],
    ])
    F2 = F_link2 * np.array([cos(phi2), sin(phi2)])
    tau2 = J2.T @ F2

    # ---- Link 3 force ----
    # Attachment: p_att3 = l1*[cos(a1),sin(a1)] + l2*[cos(a2),sin(a2)] + force_s3*[cos(a3),sin(a3)] + r_circle3*[-sin(a3), cos(a3)]
    J3 = np.array([
        [-(l1*sin(a1) + l2*sin(a2) + force_s3*sin(a3) + r_circle3*cos(a3)),  -(l2*sin(a2) + force_s3*sin(a3) + r_circle3*cos(a3)),  -(force_s3*sin(a3) + r_circle3*cos(a3))],
        [ l1*cos(a1) + l2*cos(a2) + force_s3*cos(a3) - r_circle3*sin(a3),     l2*cos(a2) + force_s3*cos(a3) - r_circle3*sin(a3),     force_s3*cos(a3) - r_circle3*sin(a3)  ],
    ])
    F3 = F_link3 * np.array([cos(phi3), sin(phi3)])
    tau3 = J3.T @ F3

    return tau1 + tau2 + tau3



def dynamics(t, state, force_s=None, r_circle=None, F_link=None, aim_frac=None):
    """
    State vector: [phi1, phi2, phi3, phi1_dot, phi2_dot, phi3_dot]
    EOM: M(q)*q_ddot + C(q,q_dot)*q_dot + Tau_K(q) + Tau_B(q_dot) = 0
    => q_ddot = M^{-1} * (-C*q_dot - Tau_K - Tau_B + tau_ext)

    force_s  : tuple (s1, s2, s3) - attachment distances along each link [m]
    r_circle : tuple (r1, r2, r3) - circle radii at the midpoints [m]
    F_link   : tuple of three floats  F_i -> float [N]
    aim_frac : tuple (aim1, aim2, aim3) - fractional target positions (0-1) on
               the link below.  Pass None to disable external forces.
    """
    th1, th2, th3, th1d, th2d, th3d = state
    q_dot = np.array([th1d, th2d, th3d])

    M_mat = M(th1, th2, th3)
    C_mat = C(th1, th2, th3, th1d, th2d, th3d)
    tau_k = Tau_K(th1, th2, th3)
    tau_b = Tau_B(th1d, th2d, th3d)
    if force_s is not None:
        tau_ext = tau_link_forces(th1, th2, th3, t, state, force_s, r_circle, F_link, aim_frac)
    else:
        tau_ext = np.zeros(3)

    rhs = -C_mat @ q_dot - tau_k - tau_b + tau_ext
    q_ddot = np.linalg.solve(M_mat, rhs)
    return [th1d, th2d, th3d, q_ddot[0], q_ddot[1], q_ddot[2]]


def cable_tensions(state, force_s, r_circle, aim_frac):
    """
    Compute the cable tension each line would experience from the finger's
    passive spring and damping torques alone.

    Physical meaning
    ----------------
    The passive torques tau_passive = Tau_K(q) + Tau_B(q_dot) tend to return
    the joints to their rest angles.  If three cables are attached at the
    configured attachment points and aimed toward their respective target
    points, each cable would experience a tension T_i such that:

        G @ [T1, T2, T3] = tau_passive

    where column i of G is  g_i = J_i^T @ d_i  (generalized force per unit
    tension in cable i).  G is 3x3 upper-triangular and always invertible for
    non-degenerate configurations.

    Parameters
    ----------
    state    : array-like [phi1, phi2, phi3, phi1_dot, phi2_dot, phi3_dot]
    force_s  : tuple (s1, s2, s3) - attachment distances [m]
    r_circle : tuple (r1, r2, r3) - circle radii [m]
    aim_frac : tuple (aim1, aim2, aim3) - fractional target positions (0-1)

    Returns
    -------
    T : np.ndarray, shape (3,)   cable tensions [N]
        Positive = cable is being pulled taut toward the target.
        Negative = springs are pushing the link away (cable would be slack).
    """
    th1, th2, th3, th1d, th2d, th3d = state
    force_s1, force_s2, force_s3 = force_s
    r_circle1, r_circle2, r_circle3 = r_circle
    aim1, aim2, aim3 = aim_frac

    a1 = th1
    a2 = th1 + th2
    a3 = th1 + th2 + th3

    MCP = np.array([l1*cos(a1), l1*sin(a1)])
    PIP = MCP + np.array([l2*cos(a2), l2*sin(a2)])

    # Attachment points (same geometry as tau_link_forces)
    att1 = np.array([force_s1*cos(a1) - r_circle1*sin(a1),
                     force_s1*sin(a1) + r_circle1*cos(a1)])
    att2 = MCP + np.array([force_s2*cos(a2) - r_circle2*sin(a2),
                           force_s2*sin(a2) + r_circle2*cos(a2)])
    att3 = PIP + np.array([force_s3*cos(a3) - r_circle3*sin(a3),
                           force_s3*sin(a3) + r_circle3*cos(a3)])

    # Target points on the link below
    target1 = np.array([-aim1 * l0, 0.0])
    target2 = aim2 * l1 * np.array([cos(a1), sin(a1)])
    target3 = MCP + aim3 * l2 * np.array([cos(a2), sin(a2)])

    # Unit direction vectors from attachment toward target
    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else np.zeros(2)

    d1 = _unit(target1 - att1)
    d2 = _unit(target2 - att2)
    d3 = _unit(target3 - att3)

    # Jacobians (identical to those in tau_link_forces)
    J1 = np.array([
        [-(force_s1*sin(a1) + r_circle1*cos(a1)),  0.0,  0.0],
        [ force_s1*cos(a1) - r_circle1*sin(a1),    0.0,  0.0],
    ])
    J2 = np.array([
        [-(l1*sin(a1) + force_s2*sin(a2) + r_circle2*cos(a2)),  -(force_s2*sin(a2) + r_circle2*cos(a2)),  0.0],
        [ l1*cos(a1) + force_s2*cos(a2) - r_circle2*sin(a2),     force_s2*cos(a2) - r_circle2*sin(a2),   0.0],
    ])
    J3 = np.array([
        [-(l1*sin(a1) + l2*sin(a2) + force_s3*sin(a3) + r_circle3*cos(a3)),  -(l2*sin(a2) + force_s3*sin(a3) + r_circle3*cos(a3)),  -(force_s3*sin(a3) + r_circle3*cos(a3))],
        [ l1*cos(a1) + l2*cos(a2) + force_s3*cos(a3) - r_circle3*sin(a3),     l2*cos(a2) + force_s3*cos(a3) - r_circle3*sin(a3),     force_s3*cos(a3) - r_circle3*sin(a3)  ],
    ])

    # G[:,i] = generalized force per unit tension in cable i
    G = np.column_stack([J1.T @ d1, J2.T @ d2, J3.T @ d3])

    # Passive torques the cables must balance
    tau_passive = Tau_K(th1, th2, th3) + Tau_B(th1d, th2d, th3d)

    try:
        T = np.linalg.solve(G, tau_passive)
    except np.linalg.LinAlgError:
        T = np.full(3, np.nan)
    return T