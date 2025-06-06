import numpy as np
from scipy.linalg import logm, expm
import time
import sys
from scipy.spatial.transform import Rotation as R
#from python_pkg.jacobian import compute_jacobian
import logging
logging.basicConfig(level=logging.WARNING)
import pandas as pd



class TrajectoryGenerator():
    def __init__(self, sampling_frequency=100, traj_time = 20, traj_type="straight", dh_l=[0.10555,0.176,0.3,0.32,0.2251]):
        self.traj_time = traj_time
        self.traj_type = traj_type
        self.traj_status = True
        self.dh_l = dh_l
        self.sampling_frequency = sampling_frequency
        l1,l2, l3, l4, l5 = self.dh_l[0],self.dh_l[1],self.dh_l[2],self.dh_l[3],self.dh_l[4]

        ##### trajectory flags
        self.grip_reach_f = False
        self.put_reach_f = False

        #### null space
        self.max_joints_limit = np.array([3.14,1.56,3.141,2.446,3.141,1.306,6.28,3.14,1.56,3.141,2.446,3.141,1.306,6.28])
        self.min_joints_limit = np.array([-3.141,-1.56,-3.141,-1.365,-3.141,-1.306,-6.28,-3.141,-1.56,-3.141,-1.365,-3.141,-1.306,-6.28])
        self.mean_joints_limit = (self.max_joints_limit + self.min_joints_limit)/2
        self.max_min_dif_sq = (self.max_joints_limit-self.min_joints_limit)**2
        self.null_space = True
        
        self.R_prev = None
        ### traj generator
        self.prev_position = None
        self.prev_euler = None

        ### click
        self.joint_states = None #[3.14, -1.5699755717780695, 1.57,-1.569931101434043, -1.57, 1.57]
        self.posedot = None
        self.pose = None
        self.pose_a = None
        self.joint_states_prev = None #[3.14,-1.5699755717780695, 1.57,-1.569931101434043, -1.57, 1.57]
        self.joint_angles_both = None

    def rot2eul(self,rot_matrix, seq='XYZ'):
        """
        Convert rotation matrix to Euler angles.
        Uses scipy's spatial transform for high efficiency.
        """
        rotation = R.from_matrix(rot_matrix)
        eul = rotation.as_euler(seq, degrees=False)
        return eul
    
    def eul2angular_vel(self,euler):
        alpha, beta, gamma, alpha_dot, beta_dot, gamma_dot = euler

        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)

        omega_x = ((ca * sg + cg * sa * sb) * 
                   (ca * sg * alpha_dot + cg * sa * gamma_dot - ca * cb * cg * beta_dot + 
                    cg * sa * sb * alpha_dot + ca * sb * sg * gamma_dot) +(ca * cg - sa * sb * sg) * 
                    (ca * cg * alpha_dot - sa * sg * gamma_dot + ca * cb * sg * beta_dot +
                    ca * cg * sb * gamma_dot - sa * sb * sg * alpha_dot) +
                    cb * sa * (cb * sa * alpha_dot + ca * sb * beta_dot))

        omega_y = ca * beta_dot - cb * sa * gamma_dot
        omega_z = sa * beta_dot + ca * cb * gamma_dot

        return np.array([omega_x, omega_y, omega_z])
    
    def compute_differentiation(self, prev, curr, dt):
        """
        Compute linear velocity given previous pose, current pose, and time increment.

        Args:
            prev (np.array): Previous position (x, y, z)
            curr (np.array): Current position (x, y, z)
            dt (float): Time increment (should be > 0)

        Returns:
            np.array: Linear velocity (vx, vy, vz)
        """
        if dt <= 0:
            raise ValueError("Time increment must be positive.")
    
        velocity = (curr - prev) / dt
        return velocity
    
    ###################################################################
    def compute_jacobian(self, theta, dh_l):
        theta = np.array(theta)
        l1,l2,l3,l4,l5 = dh_l[0], dh_l[1], dh_l[2], dh_l[3], dh_l[4]
        
        t1, t2, t3, t4, t5, t6 = theta[0] + np.pi / 2, theta[1], theta[2], theta[3], theta[4], theta[5]
      
        sin_t1 = np.sin(t1)
        cos_t1 = np.cos(t1)
        sin_t2 = np.sin(t2)
        cos_t2 = np.cos(t2)
        sin_t3 = np.sin(t3)
        cos_t3 = np.cos(t3)
        sin_t4 = np.sin(t4)
        cos_t4 = np.cos(t4)
        sin_t5 = np.sin(t5)
        cos_t5 = np.cos(t5)
        sin_t6 = np.sin(t6)
        cos_t6 = np.cos(t6)

        expr1 = sin_t1
        expr2 = -cos_t1 * sin_t2
        expr3 = cos_t3 * sin_t1 + cos_t2 * cos_t1 * sin_t3
        expr4 = (
            sin_t4 * (sin_t3 * sin_t1 - cos_t2 * cos_t3 * cos_t1)
            - cos_t4 * cos_t1 * sin_t2
        )
        expr5 = (
            cos_t5 * (cos_t3 * sin_t1 + cos_t2 * cos_t1 * sin_t3)
            - sin_t5 * (
                cos_t4 * (sin_t3 * sin_t1 - cos_t2 * cos_t3 * cos_t1)
                + cos_t1 * sin_t2 * sin_t4
            )
        )
        expr6 = (
            cos_t6 * (
                sin_t4 * (sin_t3 * sin_t1 - cos_t2 * cos_t3 * cos_t1)
                - cos_t4 * cos_t1 * sin_t2
            )
            + sin_t6 * (
                cos_t5 * (
                    cos_t4 * (sin_t3 * sin_t1 - cos_t2 * cos_t3 * cos_t1)
                    + cos_t1 * sin_t2 * sin_t4
                )
                + sin_t5 * (cos_t3 * sin_t1 + cos_t2 * cos_t1 * sin_t3)
            )
        )

        ####row 2
        expr21 = -1
        expr22 = 0
        expr23 = -cos_t2
        expr24 = -sin_t2 * sin_t3
        expr25 = cos_t3 * sin_t2 * sin_t4 - cos_t2 * cos_t4
        expr26 = -sin_t5 * (cos_t2 * sin_t4 + cos_t3 * cos_t4 * sin_t2) - cos_t5 * sin_t2 * sin_t3
        expr27 = (
            sin_t6 * (cos_t5 * (cos_t2 * sin_t4 + cos_t3 * cos_t4 * sin_t2) - sin_t2 * sin_t3 * sin_t5)
            - cos_t6 * (cos_t2 * cos_t4 - cos_t3 * sin_t2 * sin_t4)
        )

        #### row3
        expr31 = 0
        expr32 = -cos_t1
        expr33 = -sin_t2 * sin_t1
        expr34 = cos_t2 * sin_t3 * sin_t1 - cos_t3 * cos_t1
        expr35 = -sin_t4 * (cos_t1 * sin_t3 + cos_t2 * cos_t3 * sin_t1) - cos_t4 * sin_t2 * sin_t1
        expr36 = (
            sin_t5 * (
                cos_t4 * (cos_t1 * sin_t3 + cos_t2 * cos_t3 * sin_t1)
                - sin_t2 * sin_t4 * sin_t1
            )
            - cos_t5 * (cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1)
        )
        expr37 = (
            -sin_t6 * (
                cos_t5 * (
                    cos_t4 * (cos_t1 * sin_t3 + cos_t2 * cos_t3 * sin_t1)
                    - sin_t2 * sin_t4 * sin_t1
                )
                + sin_t5 * (cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1)
            )
            - cos_t6 * (
                sin_t4 * (cos_t1 * sin_t3 + cos_t2 * cos_t3 * sin_t1)
                + cos_t4 * sin_t2 * sin_t1
            )
        )

        #### row4
        A = cos_t1 * sin_t3 + cos_t2 * cos_t3 * sin_t1
        B = cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1
        C = sin_t4 * A + cos_t4 * sin_t2 * sin_t1
        D = cos_t4 * A - sin_t2 * sin_t4 * sin_t1
        E = cos_t3 * sin_t1 + cos_t2 * cos_t1 * sin_t3
        F = sin_t3 * sin_t1 - cos_t2 * cos_t3 * cos_t1
        G = sin_t4 * F - cos_t4 * cos_t1 * sin_t2
        H = cos_t4 * F + cos_t1 * sin_t2 * sin_t4
        expr41 = l4 * C + l5 * (sin_t6 * (cos_t5 * D + sin_t5 * B) + cos_t6 * C) + l3 * sin_t2 * sin_t1
        expr42 = l5 * (
            sin_t6 * (
                cos_t5 * (cos_t2 * cos_t1 * sin_t4 + cos_t3 * cos_t4 * cos_t1 * sin_t2)
                - cos_t1 * sin_t2 * sin_t3 * sin_t5
            )
            - cos_t6 * (cos_t2 * cos_t4 * cos_t1 - cos_t3 * cos_t1 * sin_t2 * sin_t4)
        ) - l4 * (cos_t2 * cos_t4 * cos_t1 - cos_t3 * cos_t1 * sin_t2 * sin_t4) - l3 * cos_t2 * cos_t1

        expr43 = l4 * sin_t4 * E - l5 * (
            sin_t6 * (
                sin_t5 * (sin_t3 * sin_t1 - cos_t2 * cos_t3 * cos_t1)
                - cos_t4 * cos_t5 * E
            )
            - cos_t6 * sin_t4 * E
        )

        expr44 = l5 * (
            cos_t6 * (cos_t4 * F + cos_t1 * sin_t2 * sin_t4)
            - cos_t5 * sin_t6 * (sin_t4 * F - cos_t4 * cos_t1 * sin_t2)
        ) + l4 * (cos_t4 * F + cos_t1 * sin_t2 * sin_t4)

        expr45 = -l5 * sin_t6 * (
            sin_t5 * (cos_t4 * F + cos_t1 * sin_t2 * sin_t4)
            - cos_t5 * E
        )

        expr46 = -l5 * (
            sin_t6 * G
            - cos_t6 * (cos_t5 * H + sin_t5 * E)
        )

        expr47 = 0

        ##### row5
        """term1 = cos_t4 * sin_t2 + cos_t2 * cos_t3 * sin_t4
        term2 = sin_t2 * sin_t4 - cos_t2 * cos_t3 * cos_t4
        term3 = cos_t2 * sin_t4 + cos_t3 * cos_t4 * sin_t2
        term4 = cos_t2 * cos_t4 - cos_t3 * sin_t2 * sin_t4

        expr51 = 0.0
        expr
        # expr52 = -l5 * (sin_t6 * (cos_t3 * sin_t2 * sin_t5 + cos_t4 * cos_t5 * sin_t2 * sin_t3) + cos_t6 * sin_t2 * sin_t3 * sin_t4) - l4 * sin_t2 * sin_t3 * sin_t4
        expr53 = l4 * term3 + l5 * (cos_t6 * term3 + cos_t5 * sin_t6 * term4)
        expr54 = -l5 * sin_t6 * (sin_t5 * term3 + cos_t5 * sin_t2 * sin_t3)
        expr55 = l5 * (cos_t6 * (cos_t5 * term3 - sin_t2 * sin_t3 * sin_t5) + sin_t6 * term4)
        expr56 = 0
        expr57 = 0
        """
        expr51 = 0
        expr52 = l4 * (np.cos(theta[3]) * np.sin(theta[1]) + np.cos(theta[1]) * np.cos(theta[2]) * np.sin(theta[3]))- l5 * (
        np.sin(theta[5]) * (
            np.cos(theta[4]) * (np.sin(theta[1]) * np.sin(theta[3]) - np.cos(theta[1]) * np.cos(theta[2]) * np.cos(theta[3]))
            + np.cos(theta[1]) * np.sin(theta[2]) * np.sin(theta[4])
        )
        - np.cos(theta[5]) * (
            np.cos(theta[3]) * np.sin(theta[1]) + np.cos(theta[1]) * np.cos(theta[2]) * np.sin(theta[3])
        ))+ l3 * np.sin(theta[1])

        expr53 = - l5 * (
        np.sin(theta[5]) * (
            np.cos(theta[2]) * np.sin(theta[1]) * np.sin(theta[4])
            + np.cos(theta[3]) * np.cos(theta[4]) * np.sin(theta[1]) * np.sin(theta[2])
        )
        + np.cos(theta[5]) * np.sin(theta[1]) * np.sin(theta[2]) * np.sin(theta[3]))- l4 * np.sin(theta[1]) * np.sin(theta[2]) * np.sin(theta[3])

        expr54 =l4 * (
        np.cos(theta[1]) * np.sin(theta[3]) + np.cos(theta[2]) * np.cos(theta[3]) * np.sin(theta[1]))+ l5 * (
        np.cos(theta[5]) * (
            np.cos(theta[1]) * np.sin(theta[3]) + np.cos(theta[2]) * np.cos(theta[3]) * np.sin(theta[1])
        )
        + np.cos(theta[4]) * np.sin(theta[5]) * (
            np.cos(theta[1]) * np.cos(theta[3]) - np.cos(theta[2]) * np.sin(theta[1]) * np.sin(theta[3])
        ))

        expr55 = -l5 * np.sin(theta[5]) * (
        np.sin(theta[4]) * (
            np.cos(theta[1]) * np.sin(theta[3]) + np.cos(theta[2]) * np.cos(theta[3]) * np.sin(theta[1])
        )
        + np.cos(theta[4]) * np.sin(theta[1]) * np.sin(theta[2]))

        expr56 = l5 * (
        np.cos(theta[5]) * (
            np.cos(theta[4]) * (
                np.cos(theta[1]) * np.sin(theta[3]) + np.cos(theta[2]) * np.cos(theta[3]) * np.sin(theta[1])
            )
            - np.sin(theta[1]) * np.sin(theta[2]) * np.sin(theta[4])
        )
        + np.sin(theta[5]) * (
            np.cos(theta[1]) * np.cos(theta[3]) - np.cos(theta[2]) * np.sin(theta[1]) * np.sin(theta[3])
        ))
        expr57 = 0.0
        #### row6
        A = sin_t3 * sin_t1 - cos_t2 * cos_t3 * cos_t1
        B = cos_t4 * cos_t1 * sin_t2
        C = sin_t4 * A - B
        D = cos_t4 * A + cos_t1 * sin_t2 * sin_t4
        E = cos_t3 * sin_t1 + cos_t2 * cos_t1 * sin_t3
        F = cos_t2 * sin_t4 * sin_t1 + cos_t3 * cos_t4 * sin_t2 * sin_t1
        G = cos_t2 * cos_t4 * sin_t1 - cos_t3 * sin_t2 * sin_t4 * sin_t1
        H = cos_t1 * sin_t3 + cos_t2 * cos_t3 * sin_t1
        I = sin_t2 * sin_t4 * sin_t1
        expr61 = (
            l4 * C
            + l5 * (cos_t6 * C + sin_t6 * (cos_t5 * D + sin_t5 * E))
            - l3 * cos_t1 * sin_t2
        )
        expr62 = (
            l5
            * (
                sin_t6
                * (
                    cos_t5 * F
                    - sin_t2 * sin_t3 * sin_t5 * sin_t1
                )
                - cos_t6 * G
            )
            - l4 * G
            - l3 * cos_t2 * sin_t1
        )
        expr63 = (
            l5
            * (
                sin_t6
                * (
                    sin_t5 * H
                    - cos_t4 * cos_t5 * (cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1)
                )
                - cos_t6 * sin_t4 * (cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1)
            )
            - l4 * sin_t4 * (cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1)
        )
        expr64 = -(
            l4
            * (
                cos_t4 * H
                - sin_t2 * sin_t4 * sin_t1
            )
            + l5
            * (
                cos_t6
                * (
                    cos_t4 * H
                    - sin_t2 * sin_t4 * sin_t1
                )
                - cos_t5 * sin_t6
                * (
                    sin_t4 * H
                    + cos_t4 * sin_t2 * sin_t1
                )
            )
        )

        expr65 = (
            l5
            * sin_t6
            * (
                sin_t5
                * (
                    cos_t4 * H
                    - sin_t2 * sin_t4 * sin_t1
                )
                - cos_t5 * (cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1)
            )
        )

        expr66 = (
            l5
            * (
                sin_t6
                * (
                    sin_t4 * H
                    + cos_t4 * sin_t2 * sin_t1
                )
                - cos_t6
                * (
                    cos_t5
                    * (
                        cos_t4 * H
                        - sin_t2 * sin_t4 * sin_t1
                    )
                    + sin_t5 * (cos_t3 * cos_t1 - cos_t2 * sin_t3 * sin_t1)
                )
            )
        )

        expr67 = 0
        return np.array([[0,expr1, expr2, expr3, expr4, expr5, expr6],
                            [expr21, expr22, expr23, expr24, expr25, expr26,expr27],
                            [expr31, expr32, expr33, expr34, expr35, expr36,expr37],
                            [expr41, expr42, expr43, expr44, expr45, expr46,expr47],
                            [expr51, expr52, expr53, expr54, expr55, expr56,expr57],
                            [expr61, expr62, expr63, expr64, expr65, expr66,expr67]])
    
    def integrate_joint_positions(self,thetadot, theta_prev, dt):
        theta_new = theta_prev + thetadot * dt
        return theta_new
    
    ########################### fk
    def t_matrix(self, a, alpha, d, theta):
        """
        Function to calculate the Homogeneous Transformation Matrix using DH parameters.
        Input: a, alpha, d, theta (DH parameters)
        Output: 4x4 Homogeneous Transformation Matrix
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
    
        T = np.array([[ct, -st,  0, a],
                  [st*ca,  ct*ca, -sa, -d*sa],
                  [st*sa,      ct*sa,     ca,    d*ca],
                  [0,       0,      0,    1]])
        return T
    
    def FK(self, theta, dh_l):
        """
        Function to calculate End-effector position and Euler angles
        Input: theta - 6x1 joint angles
        Output: out - [euler angles (XYZ); position (x,y,z)]
        """
        t1, t2, t3, t4, t5, t6, t7 = np.array([theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6]])
        l1,l2, l3, l4, l5 = dh_l[0], dh_l[1], dh_l[2], dh_l[3], dh_l[4] #np.array([0.10555,0.176,0.3,0.32,0.2251])
        dh = np.array([
            [0,        np.pi/2,      l1,      t1+np.pi/2],
            [0,        0,            l2,      0],
            [0,        np.pi/2,      0,      t2],
            [0,        -np.pi/2,     l3,      t3],
            [0,        np.pi/2,      0,      t4],
            [0,        -np.pi/2,     l4,      t5],
            [0,        np.pi/2,      0,      t6],
            [0,        -np.pi/2,     l5,      t7],
        ])
    
        # Homogeneous Transformation Matrices
        T1 = self.t_matrix(dh[0,0], dh[0,1], dh[0,2], dh[0,3])
        T1d = self.t_matrix(dh[1,0], dh[1,1], dh[1,2], dh[1,3])
        T2 = self.t_matrix(dh[2,0], dh[2,1], dh[2,2], dh[2,3])
        T3 = self.t_matrix(dh[3,0], dh[3,1], dh[3,2], dh[3,3])
        T4 = self.t_matrix(dh[4,0], dh[4,1], dh[4,2], dh[4,3])
        T5 = self.t_matrix(dh[5,0], dh[5,1], dh[5,2], dh[5,3])
        T6 = self.t_matrix(dh[6,0], dh[6,1], dh[6,2], dh[6,3])
        T7 = self.t_matrix(dh[7,0], dh[7,1], dh[7,2], dh[7,3])
        T = T1 @T1d @ T2 @ T3 @ T4 @ T5 @ T6 @T7
        angles = self.rot2eul(T[:3,:3])
        position = T[:3,3]
        return np.concatenate((angles, position)), T


    def joints_trajectory(self, t, theta, dh_l, T_init, T_final):
        """ Generate smooth trajectory using the exponential matrix method. """
        if t>self.traj_time:
            t = self.traj_time
            if self.traj_status:
                print(f"---------------trajectory execution complete.-----------------")
                self.traj_status = False
            return self.joint_angles_both

        R_init = T_init[:3, :3]
        R_final = T_final[:3, :3]
        position_init = T_init[:3, 3]
        position_final = T_final[:3, 3]

        s = 10*(t/self.traj_time)**3-15*(t/self.traj_time)**4+6*(t/self.traj_time)**5

        # Compute interpolated rotation using exponential map
        if self.R_prev is None:
            self.R_prev = R_init

        R_instant = R_init @ expm(logm(R_init.T @ R_final) * s)

        R_dot = (R_instant - self.R_prev)/(1/self.sampling_frequency)
        omega_hat = R_dot @ R_instant.T
        orientation_dot = np.array([omega_hat[2,1], omega_hat[0,2], omega_hat[1,0]])
        self.R_prev = R_instant
        
        if self.traj_type =="straight":
            position_instant = position_init + s * (position_final - position_init)
        
        elif self.traj_type == "ellipsoid":
            a = 0.3 #major
            b = 0.1 #minor
            x = position_init[0] + b*np.sin(2*np.pi*s)
            y = position_init[1]-a + a*np.cos(2*np.pi*s)
            z = position_init[2]
            position_instant = np.array([x,y,z])
        else:
            logging.warning("Provide valid trajectory typr -> traj_type='straight' or 'ellipsoid'!")
            return

        euler_instant = self.rot2eul(R_instant)

        if self.prev_position is None:
            self.prev_position = position_init
        position_dot = self.compute_differentiation(self.prev_position, position_instant, 1/self.sampling_frequency)
        
        if self.prev_euler is None:
            self.prev_euler = self.rot2eul(R_init)

        self.prev_position, self.prev_euler = position_instant, euler_instant

        pose, posedot = np.hstack([euler_instant, position_instant]), np.hstack([orientation_dot, position_dot])

        #### ik
        J = self.compute_jacobian(theta, dh_l) # 6x7
        JJT = J.T @ J    ## 7x7
        JJT_ = JJT + 0.001*np.eye(7)
        if np.linalg.det(JJT_) != 0:
            JJT_inv = np.linalg.inv(JJT_) @ J.T
        else:
            JJT_inv = np.linalg.pinv(JJT_) @ J.T 

        pose_a, _ = self.FK(theta, dh_l)   # [position, orientation]
        if pose_a is None:
            logging.warning("Either didn't receive joint states or error in FK calculation->pose_actual:None")
            return
        
        #### Null space condition
        q_dot_constraints = (self.mean_joints_limit[7:14].reshape(7,1)-np.array(theta).reshape(7,1))/self.max_min_dif_sq[7:14].reshape(7,1)
        if self.null_space:
            joint_states_dot_inst = JJT_inv @ (np.array(posedot).reshape(6,1)) + (np.eye(7)-JJT_inv@J)@ q_dot_constraints
        else:
            joint_states_dot_inst = JJT_inv @ (np.array(posedot).reshape(6,1))
        if self.joint_states_prev is None:
            print("previous pose is none")
            self.joint_states_prev = theta
        joint_states_desired = self.integrate_joint_positions(np.array(joint_states_dot_inst).flatten(), self.joint_states_prev, 1/self.sampling_frequency)
        joint_angles_both = np.concatenate((joint_states_desired, joint_states_desired))
        self.joint_states_prev = joint_states_desired
        self.joint_angles_both = joint_angles_both
        if t==0:
            print(f"-------------Executing trajectory-------------------")
            print(f"Final Transformation: {T_final}")
        return joint_angles_both
        
    def get_joints(self, t, theta, dh_l, T_init, T_final):
        return self.joints_trajectory(t, theta, dh_l, T_init, T_final)
    
if __name__== "__main__":
    obj = TrajectoryGenerator()
    _, T_init = obj.FK([-0.085, 1.396, -1.317, -1.365, 0.583, 0.221, -0.365], obj.dh_l)
    T_final = np.array([[0, -1, 0, 0], 
                                 [0, 0, -1, -.10555-0.176-0.3-0.32-0.2251], #0.10555,0.176,0.3,0.32,0.2251
                                 [1, 0, 0, 0], 
                                 [0, 0, 0, 1]])
    print(obj.get_joints(0.09, [0.1 for _ in range(7)], obj.dh_l, T_init, T_final))
    #print(obj.FK([-0.08567228792299453, 1.3969787867238597, -1.317937334925575, -1.365683925938372, 0.5836437934997851, 0.22152439148382275, -0.3658784425167152], obj.dh_l)) #9.99
    

    #print(obj.compute_jacobian([0.001543749,	1.100873737,	-1.608122752,	-1.491940354,	0.622086298,	-0.084254206,	-0.145626394], obj.dh_l)) ##9.81
    #print(obj.compute_jacobian([0.002194408,	0.00158479,	-0.001932682,	-0.000220277,	-0.001782932,	-0.001366595,	0.001518298], obj.dh_l)) ##7


    """traj = np.array(pd.read_csv("/home/cstar/Documents/dual_arm_ws/src/python_pkg/data/Book1.csv"))
    #print(f"traj: {traj[1][1]}")
    for i in range(len(traj)):
        print(f"time: {traj[i]}")
        #[0,0.002400,0.0,0.00300,0.0,0.002400,0.0]
        print(obj.get_joints(traj[i][0], traj[i][1:], obj.dh_l)) """
