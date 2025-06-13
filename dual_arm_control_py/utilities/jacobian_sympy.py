# Your symbolic Jacobian generation code (as shared above)
import sympy as sp

# Define symbols
theta = sp.symbols('theta1:8')
d = sp.symbols('d1:8')
a = sp.symbols('a1:8')
alpha = sp.symbols('alpha1:8')
l = sp.symbols('l1:6')

# DH Transformation
def dh_transform(a, alpha, d, theta):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta),  0, a],
        [sp.sin(theta)*sp.cos(alpha),  sp.cos(theta)*sp.cos(alpha), -sp.sin(alpha), -d*sp.sin(alpha)],
        [sp.sin(theta)*sp.sin(alpha),  sp.cos(theta)*sp.sin(alpha),  sp.cos(alpha), d*sp.cos(alpha)],
        [0,              0,                            0,                           1]
    ])

# Define your robot's DH parameters (example)
#l1,l2,l3,l4,l5 = 0.10555,0.176,0.3,0.32,0.2251
## right ae=rm
"""dh_params = [
    (0.0, sp.pi/2,       l[0], theta[0]+sp.pi/2),
    (0.0, 0.0,         l[1], 0.0),
    (0.0, sp.pi/2,       0.0, theta[1]),
    (0.0, -sp.pi/2,   l[2], theta[2]),
    (0.0, sp.pi/2,   0.0, theta[3]),
    (0.0,  -sp.pi/2,   l[3], theta[4]),
    (0.0,   sp.pi/2,      0.0, theta[5]),
    (0.0,   -sp.pi/2,      l[4], theta[6])
]"""

## left arm
dh_params = [
    (0.0, -sp.pi/2,       -l[0], theta[0]+sp.pi/2),
    (0.0, 0.0,         -l[1], 0.0),
    (0.0, -sp.pi/2,       0.0, theta[1]),
    (0.0, sp.pi/2,   -l[2], theta[2]),
    (0.0, -sp.pi/2,   0.0, theta[3]),
    (0.0,  sp.pi/2,   -l[3], theta[4]),
    (0.0,   -sp.pi/2,      0.0, theta[5]),
    (0.0,   sp.pi/2,      -l[4], theta[6])
]

# Forward kinematics
T01 = dh_transform(*dh_params[0])
T1d = dh_transform(*dh_params[1])
Td2 = dh_transform(*dh_params[2]) 
T23 = dh_transform(*dh_params[3])
T34 = dh_transform(*dh_params[4])
T45 = dh_transform(*dh_params[5])
T56 = dh_transform(*dh_params[6])
T67 = dh_transform(*dh_params[7])


T02 = T01@T1d@Td2
T03 = T01@T1d@Td2@T23
T04 = T01@T1d@Td2@T23@T34
T05 = T01@T1d@Td2@T23@T34@T45
T06 = T01@T1d@Td2@T23@T34@T45@T56
T07 = T01@T1d@Td2@T23@T34@T45@T56@T67

#J = sp.zeros(6, 7)

Jw = sp.Matrix.hstack(T01[0:3,2], T02[0:3,2], T03[0:3,2], T04[0:3,2], T05[0:3,2], T06[0:3,2], T07[0:3,2])  
Jv = sp.Matrix.vstack(sp.Matrix([[sp.diff(T07[0, 3], th) for th in theta]]), sp.Matrix([[sp.diff(T07[1, 3], th) for th in theta]]), sp.Matrix([[sp.diff(T07[2, 3], th) for th in theta]]))
J = sp.Matrix.vstack(Jw, Jv)
print(f"partial: {J}")


"""T = sp.eye(4)
T_list = [T]
for i in range(len(dh_params)):
    T_i = dh_transform(*dh_params[i])
    T = T * T_i
    T_list.append(T)

# Jacobian
o_n = T[:3, 3]
J = sp.zeros(6, 7)
for i in range(7):
    T_i = T_list[i]
    z_i = T_i[:3, 2]
    o_i = T_i[:3, 3]
    J[:3, i] = z_i.cross(o_n - o_i)
    J[3:, i] = z_i

print(f"jacobian: {J}")

import numpy as np

# Joint angles in radians (example)
joint_vals = {
    theta[0]: -0.08567228792299453,
    theta[1]: 1.3969787867238597,
    theta[2]: -1.317937334925575,
    theta[3]: -1.365683925938372,
    theta[4]: 0.5836437934997851,
    theta[5]: 0.22152439148382275,
    theta[6]: -0.3658784425167152
}

"""
"""joint_configs = [
    [0.0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, -np.pi/6, np.pi/8],
    [0.1, np.pi/8, np.pi/6, np.pi/4, np.pi/3, -np.pi/8, np.pi/10],
    [0.2, np.pi/7, np.pi/5, np.pi/6, np.pi/4, -np.pi/7, np.pi/12]
]

np.set_printoptions(precision=4, suppress=True)

for idx, joint_values in enumerate(joint_configs):
    joint_vals_dict = {theta[i]: joint_values[i] for i in range(7)}
    J_eval = J.subs(joint_vals_dict).evalf()
    #J_numpy = np.array(J_eval.tolist()).astype(np.float64)

    print(f"\nJacobian for configuration {idx + 1}:")
    print(J_eval)
"""



"""# Evaluate and convert to numpy array
J_eval = J.subs(joint_vals).evalf()
#J_numpy = np.array(J_eval.tolist()).astype(np.float64)

print("Numerical Jacobian (6x7):")
print(J_eval)
"""
