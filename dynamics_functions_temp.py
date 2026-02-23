from numpy import sin, cos, array


def M(theta1, theta2, theta3):
    return array([
        [7*l1**2*m1/12 + l1**2*m2 + l1**2*m3 + l1*l2*m2*cos(theta2) + 2*l1*l2*m3*cos(theta2) + l1*l3*m3*cos(theta2 + theta3) + l2**2*m2/4 + l2**2*m3 + l2*l3*m3*cos(theta3) + l3**2*m3/4 + m1*r1**2/4, l1*l2*m2*cos(theta2)/2 + l1*l2*m3*cos(theta2) + l1*l3*m3*cos(theta2 + theta3)/2 + l2**2*m2/4 + l2**2*m3 + l2*l3*m3*cos(theta3) + l3**2*m3/4, l3*m3*(2*l1*cos(theta2 + theta3) + 2*l2*cos(theta3) + l3)/4],
        [l1*l2*m2*cos(theta2)/2 + l1*l2*m3*cos(theta2) + l1*l3*m3*cos(theta2 + theta3)/2 + l2**2*m2/4 + l2**2*m3 + l2*l3*m3*cos(theta3) + l3**2*m3/4, 7*l2**2*m2/12 + l2**2*m3 + l2*l3*m3*cos(theta3) + l3**2*m3/4 + m2*r2**2/4, l3*m3*(2*l2*cos(theta3) + l3)/4],
        [l3*m3*(2*l1*cos(theta2 + theta3) + 2*l2*cos(theta3) + l3)/4, l3*m3*(2*l2*cos(theta3) + l3)/4, m3*(7*l3**2 + 3*r3**2)/12],
    ])


def C(theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot):
    return array([
        [l2*(-l1*theta2_dot*(m2 + 2*m3)*sin(theta2) - l3*m3*theta3_dot*sin(theta3))/2, l2*(-l1*theta1_dot*(m2 + 2*m3)*sin(theta2) - l1*theta2_dot*(m2 + 2*m3)*sin(theta2) - l3*m3*theta3_dot*sin(theta3))/2, l2*l3*m3*(-theta1_dot - theta2_dot - theta3_dot)*sin(theta3)/2],
        [l2*(l1*theta1_dot*(m2 + 2*m3)*sin(theta2) - l3*m3*theta3_dot*sin(theta3))/2, -l2*l3*m3*theta3_dot*sin(theta3)/2, l2*l3*m3*(-theta1_dot - theta2_dot - theta3_dot)*sin(theta3)/2],
        [l2*l3*m3*(theta1_dot + theta2_dot)*sin(theta3)/2, l2*l3*m3*(theta1_dot + theta2_dot)*sin(theta3)/2, 0],
    ])
