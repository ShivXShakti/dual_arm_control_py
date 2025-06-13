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
    def __init__(self, sampling_frequency=100, dh_l=[0.10555,0.176,0.3,0.32,0.2251],
                max_joints_limit=np.array([3.14,1.56,3.141,2.446,3.141,1.306,6.28,3.14,1.56,3.141,2.446,3.141,1.306,6.28]), min_joints_limit = np.array([-3.141,-1.56,-3.141,-1.365,-3.141,-1.306,-6.28,-3.141,-1.56,-3.141,-1.365,-3.141,-1.306,-6.28])):
        self.traj_status = True
        self.dh_l = dh_l
        self.sampling_frequency = sampling_frequency
        #------- trajectory execution flags -------
        self.grip_reach_f = False
        self.put_reach_f = False
        #-------- null space constraints----------
        self.max_joints_limit = max_joints_limit
        self.min_joints_limit = min_joints_limit
        self.mean_joints_limit = (self.max_joints_limit + self.min_joints_limit)/2
        self.max_min_dif_sq = (self.max_joints_limit-self.min_joints_limit)**2
        self.null_space = True
        #------- traj generator --------
        self.R_prev = None
        self.prev_position = None
        self.prev_euler = None
        #-------- click ----------
        self.joint_states = None #[3.14, -1.5699755717780695, 1.57,-1.569931101434043, -1.57, 1.57]
        self.posedot = None
        self.pose = None
        self.pose_a = None
        self.joint_states_prev = None #[3.14,-1.5699755717780695, 1.57,-1.569931101434043, -1.57, 1.57]
        self.joint_angles_both = None
    
    #---------- Methods ---------------
    def rot2eul(self,rot_matrix, seq='XYZ'):
        """
        Convert rotation matrix to Euler angles.
        Uses scipy's spatial transform for high efficiency.
        """
        rotation = R.from_matrix(rot_matrix)
        eul = rotation.as_euler(seq, degrees=False)
        return eul
    
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
    
    def compute_jacobian(self, theta, dh_l, leftarm_f=False, rightarm_f=True):
        theta = np.array(theta)
        #print(theta)
        #print(theta.shape)
        l1,l2,l3,l4,l5 = dh_l[0], dh_l[1], dh_l[2], dh_l[3], dh_l[4]
        theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, theta10, theta11, theta12, theta13, theta14 = theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7], theta[8], theta[9], theta[10], theta[11], theta[12], theta[13]

        if leftarm_f:
            J = np.array([
                [0, -np.cos(theta1), -np.sin(theta1)*np.sin(theta2), np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3), (-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.sin(theta4) - np.sin(theta1)*np.sin(theta2)*np.cos(theta4), -((-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.cos(theta4) + np.sin(theta1)*np.sin(theta2)*np.sin(theta4))*np.sin(theta5) + (np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3))*np.cos(theta5), (((-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.cos(theta4) + np.sin(theta1)*np.sin(theta2)*np.sin(theta4))*np.cos(theta5) + (np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3))*np.sin(theta5))*np.sin(theta6) + ((-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.sin(theta4) - np.sin(theta1)*np.sin(theta2)*np.cos(theta4))*np.cos(theta6)],
                [1, 0, np.cos(theta2), np.sin(theta2)*np.sin(theta3), -np.sin(theta2)*np.sin(theta4)*np.cos(theta3) + np.cos(theta2)*np.cos(theta4), -(-np.sin(theta2)*np.cos(theta3)*np.cos(theta4) - np.sin(theta4)*np.cos(theta2))*np.sin(theta5) + np.sin(theta2)*np.sin(theta3)*np.cos(theta5), ((-np.sin(theta2)*np.cos(theta3)*np.cos(theta4) - np.sin(theta4)*np.cos(theta2))*np.cos(theta5) + np.sin(theta2)*np.sin(theta3)*np.sin(theta5))*np.sin(theta6) + (-np.sin(theta2)*np.sin(theta4)*np.cos(theta3) + np.cos(theta2)*np.cos(theta4))*np.cos(theta6)],
                [0, np.sin(theta1), -np.sin(theta2)*np.cos(theta1), np.sin(theta1)*np.cos(theta3) + np.sin(theta3)*np.cos(theta1)*np.cos(theta2), (np.sin(theta1)*np.sin(theta3) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.sin(theta4) - np.sin(theta2)*np.cos(theta1)*np.cos(theta4), -((np.sin(theta1)*np.sin(theta3) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.cos(theta4) + np.sin(theta2)*np.sin(theta4)*np.cos(theta1))*np.sin(theta5) + (np.sin(theta1)*np.cos(theta3) + np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.cos(theta5), (((np.sin(theta1)*np.sin(theta3) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.cos(theta4) + np.sin(theta2)*np.sin(theta4)*np.cos(theta1))*np.cos(theta5) + (np.sin(theta1)*np.cos(theta3) + np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.sin(theta5))*np.sin(theta6) + ((np.sin(theta1)*np.sin(theta3) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.sin(theta4) - np.sin(theta2)*np.cos(theta1)*np.cos(theta4))*np.cos(theta6)],
                [l3*np.sin(theta2)*np.cos(theta1) + l4*((-np.sin(theta1)*np.sin(theta3) + np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.sin(theta4) + np.sin(theta2)*np.cos(theta1)*np.cos(theta4)) + l5*((-((np.sin(theta1)*np.sin(theta3) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.cos(theta4) + np.sin(theta2)*np.sin(theta4)*np.cos(theta1))*np.cos(theta5) - (np.sin(theta1)*np.cos(theta3) + np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.sin(theta5))*np.sin(theta6) + (-(np.sin(theta1)*np.sin(theta3) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.sin(theta4) + np.sin(theta2)*np.cos(theta1)*np.cos(theta4))*np.cos(theta6)), l3*np.sin(theta1)*np.cos(theta2) + l4*(-np.sin(theta1)*np.sin(theta2)*np.sin(theta4)*np.cos(theta3) + np.sin(theta1)*np.cos(theta2)*np.cos(theta4)) + l5*((-(np.sin(theta1)*np.sin(theta2)*np.cos(theta3)*np.cos(theta4) + np.sin(theta1)*np.sin(theta4)*np.cos(theta2))*np.cos(theta5) + np.sin(theta1)*np.sin(theta2)*np.sin(theta3)*np.sin(theta5))*np.sin(theta6) + (-np.sin(theta1)*np.sin(theta2)*np.sin(theta4)*np.cos(theta3) + np.sin(theta1)*np.cos(theta2)*np.cos(theta4))*np.cos(theta6)), l4*(-np.sin(theta1)*np.sin(theta3)*np.cos(theta2) + np.cos(theta1)*np.cos(theta3))*np.sin(theta4) + l5*((-(np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3))*np.cos(theta4)*np.cos(theta5) - (np.sin(theta1)*np.cos(theta2)*np.cos(theta3) + np.sin(theta3)*np.cos(theta1))*np.sin(theta5))*np.sin(theta6) - (np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3))*np.sin(theta4)*np.cos(theta6)), l4*((np.sin(theta1)*np.cos(theta2)*np.cos(theta3) + np.sin(theta3)*np.cos(theta1))*np.cos(theta4) - np.sin(theta1)*np.sin(theta2)*np.sin(theta4)) + l5*(-(-(-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.sin(theta4) + np.sin(theta1)*np.sin(theta2)*np.cos(theta4))*np.sin(theta6)*np.cos(theta5) + (-(-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.cos(theta4) - np.sin(theta1)*np.sin(theta2)*np.sin(theta4))*np.cos(theta6)), l5*(((-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.cos(theta4) + np.sin(theta1)*np.sin(theta2)*np.sin(theta4))*np.sin(theta5) - (np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3))*np.cos(theta5))*np.sin(theta6), l5*((-((-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.cos(theta4) + np.sin(theta1)*np.sin(theta2)*np.sin(theta4))*np.cos(theta5) - (np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3))*np.sin(theta5))*np.cos(theta6) - (-(-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.sin(theta4) + np.sin(theta1)*np.sin(theta2)*np.cos(theta4))*np.sin(theta6)), 0],
                [0, l3*np.sin(theta2) + l4*(np.sin(theta2)*np.cos(theta4) + np.sin(theta4)*np.cos(theta2)*np.cos(theta3)) + l5*((-(np.sin(theta2)*np.sin(theta4) - np.cos(theta2)*np.cos(theta3)*np.cos(theta4))*np.cos(theta5) - np.sin(theta3)*np.sin(theta5)*np.cos(theta2))*np.sin(theta6) + (np.sin(theta2)*np.cos(theta4) + np.sin(theta4)*np.cos(theta2)*np.cos(theta3))*np.cos(theta6)), -l4*np.sin(theta2)*np.sin(theta3)*np.sin(theta4) + l5*((-np.sin(theta2)*np.sin(theta3)*np.cos(theta4)*np.cos(theta5) - np.sin(theta2)*np.sin(theta5)*np.cos(theta3))*np.sin(theta6) - np.sin(theta2)*np.sin(theta3)*np.sin(theta4)*np.cos(theta6)), l4*(np.sin(theta2)*np.cos(theta3)*np.cos(theta4) + np.sin(theta4)*np.cos(theta2)) + l5*(-(np.sin(theta2)*np.sin(theta4)*np.cos(theta3) - np.cos(theta2)*np.cos(theta4))*np.sin(theta6)*np.cos(theta5) + (np.sin(theta2)*np.cos(theta3)*np.cos(theta4) + np.sin(theta4)*np.cos(theta2))*np.cos(theta6)), l5*((-np.sin(theta2)*np.cos(theta3)*np.cos(theta4) - np.sin(theta4)*np.cos(theta2))*np.sin(theta5) - np.sin(theta2)*np.sin(theta3)*np.cos(theta5))*np.sin(theta6), l5*((-(-np.sin(theta2)*np.cos(theta3)*np.cos(theta4) - np.sin(theta4)*np.cos(theta2))*np.cos(theta5) - np.sin(theta2)*np.sin(theta3)*np.sin(theta5))*np.cos(theta6) - (np.sin(theta2)*np.sin(theta4)*np.cos(theta3) - np.cos(theta2)*np.cos(theta4))*np.sin(theta6)), 0], 
                [-l3*np.sin(theta1)*np.sin(theta2) + l4*((-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.sin(theta4) - np.sin(theta1)*np.sin(theta2)*np.cos(theta4)) + l5*((-((np.sin(theta1)*np.cos(theta2)*np.cos(theta3) + np.sin(theta3)*np.cos(theta1))*np.cos(theta4) - np.sin(theta1)*np.sin(theta2)*np.sin(theta4))*np.cos(theta5) - (-np.sin(theta1)*np.sin(theta3)*np.cos(theta2) + np.cos(theta1)*np.cos(theta3))*np.sin(theta5))*np.sin(theta6) + (-(np.sin(theta1)*np.cos(theta2)*np.cos(theta3) + np.sin(theta3)*np.cos(theta1))*np.sin(theta4) - np.sin(theta1)*np.sin(theta2)*np.cos(theta4))*np.cos(theta6)), l3*np.cos(theta1)*np.cos(theta2) + l4*(-np.sin(theta2)*np.sin(theta4)*np.cos(theta1)*np.cos(theta3) + np.cos(theta1)*np.cos(theta2)*np.cos(theta4)) + l5*((-(np.sin(theta2)*np.cos(theta1)*np.cos(theta3)*np.cos(theta4) + np.sin(theta4)*np.cos(theta1)*np.cos(theta2))*np.cos(theta5) + np.sin(theta2)*np.sin(theta3)*np.sin(theta5)*np.cos(theta1))*np.sin(theta6) + (-np.sin(theta2)*np.sin(theta4)*np.cos(theta1)*np.cos(theta3) + np.cos(theta1)*np.cos(theta2)*np.cos(theta4))*np.cos(theta6)), l4*(-np.sin(theta1)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.sin(theta4) + l5*((-(-np.sin(theta1)*np.sin(theta3) + np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.sin(theta5) - (np.sin(theta1)*np.cos(theta3) + np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.cos(theta4)*np.cos(theta5))*np.sin(theta6) - (np.sin(theta1)*np.cos(theta3) + np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.sin(theta4)*np.cos(theta6)), l4*((-np.sin(theta1)*np.sin(theta3) + np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.cos(theta4) - np.sin(theta2)*np.sin(theta4)*np.cos(theta1)) + l5*(-(-(np.sin(theta1)*np.sin(theta3) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.sin(theta4) + np.sin(theta2)*np.cos(theta1)*np.cos(theta4))*np.sin(theta6)*np.cos(theta5) + (-(np.sin(theta1)*np.sin(theta3) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.cos(theta4) - np.sin(theta2)*np.sin(theta4)*np.cos(theta1))*np.cos(theta6)), l5*(((np.sin(theta1)*np.sin(theta3) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.cos(theta4) + np.sin(theta2)*np.sin(theta4)*np.cos(theta1))*np.sin(theta5) - (np.sin(theta1)*np.cos(theta3) + np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.cos(theta5))*np.sin(theta6), l5*((-((np.sin(theta1)*np.sin(theta3) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.cos(theta4) + np.sin(theta2)*np.sin(theta4)*np.cos(theta1))*np.cos(theta5) - (np.sin(theta1)*np.cos(theta3) + np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.sin(theta5))*np.cos(theta6) - (-(np.sin(theta1)*np.sin(theta3) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.sin(theta4) + np.sin(theta2)*np.cos(theta1)*np.cos(theta4))*np.sin(theta6)), 0]
            ])
        elif rightarm_f:
            J = np.array([[0, np.cos(theta8), np.sin(theta8)*np.sin(theta9), -np.sin(theta8)*np.sin(theta10)*np.cos(theta9) + np.cos(theta8)*np.cos(theta10), -(-np.sin(theta8)*np.cos(theta9)*np.cos(theta10) - np.sin(theta10)*np.cos(theta8))*np.sin(theta11) + np.sin(theta8)*np.sin(theta9)*np.cos(theta11), 
                ((-np.sin(theta8)*np.cos(theta9)*np.cos(theta10) - np.sin(theta10)*np.cos(theta8))*np.cos(theta11) + np.sin(theta8)*np.sin(theta9)*np.sin(theta11))*np.sin(theta12) + (-np.sin(theta8)*np.sin(theta10)*np.cos(theta9) + np.cos(theta8)*np.cos(theta10))*np.cos(theta12), 
                -(((-np.sin(theta8)*np.cos(theta9)*np.cos(theta10) - np.sin(theta10)*np.cos(theta8))*np.cos(theta11) + np.sin(theta8)*np.sin(theta9)*np.sin(theta11))*np.cos(theta12) - (-np.sin(theta8)*np.sin(theta10)*np.cos(theta9) + np.cos(theta8)*np.cos(theta10))*np.sin(theta12))*np.sin(theta13) + (-(-np.sin(theta8)*np.cos(theta9)*np.cos(theta10) - np.sin(theta10)*np.cos(theta8))*np.sin(theta11) + np.sin(theta8)*np.sin(theta9)*np.cos(theta11))*np.cos(theta13)],
        
                [-1, 0, -np.cos(theta9), -np.sin(theta9)*np.sin(theta10), np.sin(theta9)*np.sin(theta11)*np.cos(theta10) - np.cos(theta9)*np.cos(theta11), (-np.sin(theta9)*np.cos(theta10)*np.cos(theta11) - np.sin(theta11)*np.cos(theta9))*np.sin(theta12) - np.sin(theta9)*np.sin(theta10)*np.cos(theta12), 
                -((-np.sin(theta9)*np.cos(theta10)*np.cos(theta11) - np.sin(theta11)*np.cos(theta9))*np.cos(theta12) + np.sin(theta9)*np.sin(theta10)*np.sin(theta12))*np.sin(theta13) + (np.sin(theta9)*np.sin(theta11)*np.cos(theta10) - np.cos(theta9)*np.cos(theta11))*np.cos(theta13)], 
        
                [0, np.sin(theta8), -np.sin(theta9)*np.cos(theta8), np.sin(theta8)*np.cos(theta10) + np.sin(theta10)*np.cos(theta8)*np.cos(theta9), -(-np.sin(theta8)*np.sin(theta10) + np.cos(theta8)*np.cos(theta9)*np.cos(theta10))*np.sin(theta11) - np.sin(theta9)*np.cos(theta8)*np.cos(theta11), 
                ((-np.sin(theta8)*np.sin(theta10) + np.cos(theta8)*np.cos(theta9)*np.cos(theta10))*np.cos(theta11) - np.sin(theta9)*np.sin(theta11)*np.cos(theta8))*np.sin(theta12) + (np.sin(theta8)*np.cos(theta10) + np.sin(theta10)*np.cos(theta8)*np.cos(theta9))*np.cos(theta12), 
                -(((-np.sin(theta8)*np.sin(theta10) + np.cos(theta8)*np.cos(theta9)*np.cos(theta10))*np.cos(theta11) - np.sin(theta9)*np.sin(theta11)*np.cos(theta8))*np.cos(theta12) - (np.sin(theta8)*np.cos(theta10) + np.sin(theta10)*np.cos(theta8)*np.cos(theta9))*np.sin(theta12))*np.sin(theta13) + (-(-np.sin(theta8)*np.sin(theta10) + np.cos(theta8)*np.cos(theta9)*np.cos(theta10))*np.sin(theta11) - np.sin(theta9)*np.cos(theta8)*np.cos(theta11))*np.cos(theta13)], 
        
                [(-0.2251*((np.sin(theta8)*np.sin(theta10) - np.cos(theta8)*np.cos(theta9)*np.cos(theta10))*np.cos(theta11) + np.sin(theta9)*np.sin(theta11)*np.cos(theta8))*np.cos(theta12) - 0.2251*(np.sin(theta8)*np.cos(theta10) + np.sin(theta10)*np.cos(theta8)*np.cos(theta9))*np.sin(theta12))*np.sin(theta13) + (0.2251*(-np.sin(theta8)*np.sin(theta10) + np.cos(theta8)*np.cos(theta9)*np.cos(theta10))*np.sin(theta11) + 0.2251*np.sin(theta9)*np.cos(theta8)*np.cos(theta11))*np.cos(theta13) + (-0.32*np.sin(theta8)*np.sin(theta10) + 0.32*np.cos(theta8)*np.cos(theta9)*np.cos(theta10))*np.sin(theta11) + 0.32*np.sin(theta9)*np.cos(theta8)*np.cos(theta11) + 0.3*np.sin(theta9)*np.cos(theta8), 
                (-0.2251*(np.sin(theta8)*np.sin(theta9)*np.cos(theta10)*np.cos(theta11) + np.sin(theta8)*np.sin(theta11)*np.cos(theta9))*np.cos(theta12) + 0.2251*np.sin(theta8)*np.sin(theta9)*np.sin(theta10)*np.sin(theta12))*np.sin(theta13) + (-0.2251*np.sin(theta8)*np.sin(theta9)*np.sin(theta11)*np.cos(theta10) + 0.2251*np.sin(theta8)*np.cos(theta9)*np.cos(theta11))*np.cos(theta13) - 0.32*np.sin(theta8)*np.sin(theta9)*np.sin(theta11)*np.cos(theta10) + 0.32*np.sin(theta8)*np.cos(theta9)*np.cos(theta11) + 0.3*np.sin(theta8)*np.cos(theta9), 
                (-0.2251*(np.sin(theta8)*np.sin(theta10)*np.cos(theta9) - np.cos(theta8)*np.cos(theta10))*np.cos(theta11)*np.cos(theta12) - 0.2251*(np.sin(theta8)*np.cos(theta9)*np.cos(theta10) + np.sin(theta10)*np.cos(theta8))*np.sin(theta12))*np.sin(theta13) + 0.2251*(-np.sin(theta8)*np.sin(theta10)*np.cos(theta9) + np.cos(theta8)*np.cos(theta10))*np.sin(theta11)*np.cos(theta13) + (-0.32*np.sin(theta8)*np.sin(theta10)*np.cos(theta9) + 0.32*np.cos(theta8)*np.cos(theta10))*np.sin(theta11), 
                -0.2251*(-(-np.sin(theta8)*np.cos(theta9)*np.cos(theta10) - np.sin(theta10)*np.cos(theta8))*np.sin(theta11) + np.sin(theta8)*np.sin(theta9)*np.cos(theta11))*np.sin(theta13)*np.cos(theta12) + (0.2251*(np.sin(theta8)*np.cos(theta9)*np.cos(theta10) + np.sin(theta10)*np.cos(theta8))*np.cos(theta11) - 0.2251*np.sin(theta8)*np.sin(theta9)*np.sin(theta11))*np.cos(theta13) + (0.32*np.sin(theta8)*np.cos(theta9)*np.cos(theta10) + 0.32*np.sin(theta10)*np.cos(theta8))*np.cos(theta11) - 0.32*np.sin(theta8)*np.sin(theta9)*np.sin(theta11), 
                (0.2251*((-np.sin(theta8)*np.cos(theta9)*np.cos(theta10) - np.sin(theta10)*np.cos(theta8))*np.cos(theta11) + np.sin(theta8)*np.sin(theta9)*np.sin(theta11))*np.sin(theta12) - 0.2251*(np.sin(theta8)*np.sin(theta10)*np.cos(theta9) - np.cos(theta8)*np.cos(theta10))*np.cos(theta12))*np.sin(theta13), 
                (-0.2251*((-np.sin(theta8)*np.cos(theta9)*np.cos(theta10) - np.sin(theta10)*np.cos(theta8))*np.cos(theta11) + np.sin(theta8)*np.sin(theta9)*np.sin(theta11))*np.cos(theta12) + 0.2251*(-np.sin(theta8)*np.sin(theta10)*np.cos(theta9) + np.cos(theta8)*np.cos(theta10))*np.sin(theta12))*np.cos(theta13) - (-0.2251*(-np.sin(theta8)*np.cos(theta9)*np.cos(theta10) - np.sin(theta10)*np.cos(theta8))*np.sin(theta11) + 0.2251*np.sin(theta8)*np.sin(theta9)*np.cos(theta11))*np.sin(theta13), 0], 
        
                [0, (-0.2251*(np.sin(theta9)*np.sin(theta11) - np.cos(theta9)*np.cos(theta10)*np.cos(theta11))*np.cos(theta12) - 0.2251*np.sin(theta10)*np.sin(theta12)*np.cos(theta9))*np.sin(theta13) + (0.2251*np.sin(theta9)*np.cos(theta11) + 0.2251*np.sin(theta11)*np.cos(theta9)*np.cos(theta10))*np.cos(theta13) + 0.32*np.sin(theta9)*np.cos(theta11) + 0.3*np.sin(theta9) + 0.32*np.sin(theta11)*np.cos(theta9)*np.cos(theta10), 
                (-0.2251*np.sin(theta9)*np.sin(theta10)*np.cos(theta11)*np.cos(theta12) - 0.2251*np.sin(theta9)*np.sin(theta12)*np.cos(theta10))*np.sin(theta13) - 0.2251*np.sin(theta9)*np.sin(theta10)*np.sin(theta11)*np.cos(theta13) - 0.32*np.sin(theta9)*np.sin(theta10)*np.sin(theta11), -0.2251*(np.sin(theta9)*np.sin(theta11)*np.cos(theta10) - np.cos(theta9)*np.cos(theta11))*np.sin(theta13)*np.cos(theta12) + (0.2251*np.sin(theta9)*np.cos(theta10)*np.cos(theta11) + 0.2251*np.sin(theta11)*np.cos(theta9))*np.cos(theta13) + 0.32*np.sin(theta9)*np.cos(theta10)*np.cos(theta11) + 0.32*np.sin(theta11)*np.cos(theta9), 
                (0.2251*(-np.sin(theta9)*np.cos(theta10)*np.cos(theta11) - np.sin(theta11)*np.cos(theta9))*np.sin(theta12) - 0.2251*np.sin(theta9)*np.sin(theta10)*np.cos(theta12))*np.sin(theta13), (-0.2251*(-np.sin(theta9)*np.cos(theta10)*np.cos(theta11) - np.sin(theta11)*np.cos(theta9))*np.cos(theta12) - 0.2251*np.sin(theta9)*np.sin(theta10)*np.sin(theta12))*np.cos(theta13) - (0.2251*np.sin(theta9)*np.sin(theta11)*np.cos(theta10) - 0.2251*np.cos(theta9)*np.cos(theta11))*np.sin(theta13), 0], 
        
                [(-0.2251*((-np.sin(theta8)*np.cos(theta9)*np.cos(theta10) - np.sin(theta10)*np.cos(theta8))*np.cos(theta11) + np.sin(theta8)*np.sin(theta9)*np.sin(theta11))*np.cos(theta12) - 0.2251*(np.sin(theta8)*np.sin(theta10)*np.cos(theta9) - np.cos(theta8)*np.cos(theta10))*np.sin(theta12))*np.sin(theta13) + (0.2251*(np.sin(theta8)*np.cos(theta9)*np.cos(theta10) + np.sin(theta10)*np.cos(theta8))*np.sin(theta11) + 0.2251*np.sin(theta8)*np.sin(theta9)*np.cos(theta11))*np.cos(theta13) + (0.32*np.sin(theta8)*np.cos(theta9)*np.cos(theta10) + 0.32*np.sin(theta10)*np.cos(theta8))*np.sin(theta11) + 0.32*np.sin(theta8)*np.sin(theta9)*np.cos(theta11) + 0.3*np.sin(theta8)*np.sin(theta9),
                (-0.2251*(-np.sin(theta9)*np.cos(theta8)*np.cos(theta10)*np.cos(theta11) - np.sin(theta11)*np.cos(theta8)*np.cos(theta9))*np.cos(theta12) - 0.2251*np.sin(theta9)*np.sin(theta10)*np.sin(theta12)*np.cos(theta8))*np.sin(theta13) + (0.2251*np.sin(theta9)*np.sin(theta11)*np.cos(theta8)*np.cos(theta10) - 0.2251*np.cos(theta8)*np.cos(theta9)*np.cos(theta11))*np.cos(theta13) + 0.32*np.sin(theta9)*np.sin(theta11)*np.cos(theta8)*np.cos(theta10) - 0.32*np.cos(theta8)*np.cos(theta9)*np.cos(theta11) - 0.3*np.cos(theta8)*np.cos(theta9), 
                (-0.2251*(np.sin(theta8)*np.sin(theta10) - np.cos(theta8)*np.cos(theta9)*np.cos(theta10))*np.sin(theta12) - 0.2251*(-np.sin(theta8)*np.cos(theta10) - np.sin(theta10)*np.cos(theta8)*np.cos(theta9))*np.cos(theta11)*np.cos(theta12))*np.sin(theta13) + (0.32*np.sin(theta8)*np.cos(theta10) + 0.32*np.sin(theta10)*np.cos(theta8)*np.cos(theta9))*np.sin(theta11) + 0.2251*(np.sin(theta8)*np.cos(theta10) + np.sin(theta10)*np.cos(theta8)*np.cos(theta9))*np.sin(theta11)*np.cos(theta13), 
                -0.2251*(-(-np.sin(theta8)*np.sin(theta10) + np.cos(theta8)*np.cos(theta9)*np.cos(theta10))*np.sin(theta11) - np.sin(theta9)*np.cos(theta8)*np.cos(theta11))*np.sin(theta13)*np.cos(theta12) + (0.2251*(np.sin(theta8)*np.sin(theta10) - np.cos(theta8)*np.cos(theta9)*np.cos(theta10))*np.cos(theta11) + 0.2251*np.sin(theta9)*np.sin(theta11)*np.cos(theta8))*np.cos(theta13) + (0.32*np.sin(theta8)*np.sin(theta10) - 0.32*np.cos(theta8)*np.cos(theta9)*np.cos(theta10))*np.cos(theta11) + 0.32*np.sin(theta9)*np.sin(theta11)*np.cos(theta8), 
                (0.2251*((-np.sin(theta8)*np.sin(theta10) + np.cos(theta8)*np.cos(theta9)*np.cos(theta10))*np.cos(theta11) - np.sin(theta9)*np.sin(theta11)*np.cos(theta8))*np.sin(theta12) - 0.2251*(-np.sin(theta8)*np.cos(theta10) - np.sin(theta10)*np.cos(theta8)*np.cos(theta9))*np.cos(theta12))*np.sin(theta13), 
                (-0.2251*((-np.sin(theta8)*np.sin(theta10) + np.cos(theta8)*np.cos(theta9)*np.cos(theta10))*np.cos(theta11) - np.sin(theta9)*np.sin(theta11)*np.cos(theta8))*np.cos(theta12) + 0.2251*(np.sin(theta8)*np.cos(theta10) + np.sin(theta10)*np.cos(theta8)*np.cos(theta9))*np.sin(theta12))*np.cos(theta13) - (-0.2251*(-np.sin(theta8)*np.sin(theta10) + np.cos(theta8)*np.cos(theta9)*np.cos(theta10))*np.sin(theta11) - 0.2251*np.sin(theta9)*np.cos(theta8)*np.cos(theta11))*np.sin(theta13), 0]])
        else:
            J = np.zeros((6,7))
        return J
    
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
    
    def FK(self, theta, dh_l, leftarm_f = False, rightarm_f = True):
        """
        Function to calculate End-effector position and Euler angles
        Input: theta - 6x1 joint angles
        Output: out - [euler angles (XYZ); position (x,y,z)]
        """
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14 = theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7], theta[8], theta[9], theta[10], theta[11], theta[12], theta[13]
        l1,l2, l3, l4, l5 = dh_l[0], dh_l[1], dh_l[2], dh_l[3], dh_l[4] #np.array([0.10555,0.176,0.3,0.32,0.2251])
        if rightarm_f:
            dh_params = np.array([
            [0,        np.pi/2,      l1,      t1+np.pi/2],
            [0,        0,            l2,      0],
            [0,        np.pi/2,      0,      t2],
            [0,        -np.pi/2,     l3,      t3],
            [0,        np.pi/2,      0,      t4],
            [0,        -np.pi/2,     l4,      t5],
            [0,        np.pi/2,      0,      t6],
            [0,        -np.pi/2,     l5,      t7]
        ])
        elif leftarm_f:
            dh_params = np.array([
            [0.0, np.pi/2,       -l1, t8-np.pi/2],
            [0.0, 0.0,         -l2, 0.0],
            [0.0, -np.pi/2,       0.0, t9],
            [0.0, np.pi/2,   -l3, t10],
            [0.0, -np.pi/2,   0.0, t11],
            [0.0,  np.pi/2,   -l4, t12],
            [0.0,   -np.pi/2,      0.0, t13],
            [0.0,   np.pi/2,      -l5, t14]
        ])
        else:
            print(f"No dh parameters are given, please mention leftarm_f=True or rightarm_f=True")
            return None, None
    
        # Homogeneous Transformation Matrices
        T01 = self.t_matrix(dh_params[0,0], dh_params[0,1], dh_params[0,2], dh_params[0,3])
        T1d = self.t_matrix(dh_params[1,0], dh_params[1,1], dh_params[1,2], dh_params[1,3])
        Td2 = self.t_matrix(dh_params[2,0], dh_params[2,1], dh_params[2,2], dh_params[2,3])
        T23 = self.t_matrix(dh_params[3,0], dh_params[3,1], dh_params[3,2], dh_params[3,3])
        T34 = self.t_matrix(dh_params[4,0], dh_params[4,1], dh_params[4,2], dh_params[4,3])
        T45 = self.t_matrix(dh_params[5,0], dh_params[5,1], dh_params[5,2], dh_params[5,3])
        T56 = self.t_matrix(dh_params[6,0], dh_params[6,1], dh_params[6,2], dh_params[6,3])
        T67 = self.t_matrix(dh_params[7,0], dh_params[7,1], dh_params[7,2], dh_params[7,3])
        T = T01 @T1d @ Td2 @ T23 @ T34 @ T45 @ T56 @T67
        angles = self.rot2eul(T[:3,:3])
        position = T[:3,3]
        return np.concatenate((angles, position)), T


    def joints_trajectory(self, t, traj_time, theta, dh_l, T_init, T_final, leftarm_f, rightarm_f):
        """ Generate smooth trajectory using the exponential matrix method. """
        if t>traj_time:
            t = traj_time
            if self.traj_status:
                print(f"--------------- left:{leftarm_f}, right:{rightarm_f} arm trajectory execution complete.-----------------")
                self.traj_status = False
            return self.joint_states_prev

        R_init = T_init[:3, :3]
        R_final = T_final[:3, :3]
        position_init = T_init[:3, 3]
        position_final = T_final[:3, 3]
        s = 10*(t/traj_time)**3-15*(t/traj_time)**4+6*(t/traj_time)**5
        position_instant = position_init + s * (position_final - position_init)

        if self.R_prev is None:
                self.R_prev = R_init

        R_instant = R_init @ expm(logm(R_init.T @ R_final) * s)

        R_dot = (R_instant - self.R_prev)/(1/self.sampling_frequency)
        omega_hat = R_dot @ R_instant.T
        orientation_dot = np.array([omega_hat[2,1], omega_hat[0,2], omega_hat[1,0]])
        self.R_prev = R_instant

        euler_instant = self.rot2eul(R_instant)

        if self.prev_position is None:
            self.prev_position = position_init
        position_dot = self.compute_differentiation(self.prev_position, position_instant, 1/self.sampling_frequency)
        
        if self.prev_euler is None:
            self.prev_euler = self.rot2eul(R_init)

        self.prev_position, self.prev_euler = position_instant, euler_instant

        pose, posedot = np.hstack([euler_instant, position_instant]), np.hstack([orientation_dot, position_dot])
        
        #### Null space condition
        q_dot_constraints = (self.mean_joints_limit.reshape(14,1)-np.array(theta).reshape(14,1))/self.max_min_dif_sq.reshape(14,1)
        #### ik
        J = self.compute_jacobian(theta, dh_l, leftarm_f, rightarm_f) # 6x7
        JJT = J.T @ J    ## 7x7
        JJT_ = JJT + 0.001*np.eye(7)
        if np.linalg.det(JJT_) != 0:
            JJT_inv = np.linalg.inv(JJT_) @ J.T
        else:
            JJT_inv = np.linalg.pinv(JJT_) @ J.T 

        pose_a, _ = self.FK(theta, dh_l, leftarm_f, rightarm_f)   # [position, orientation]
        if pose_a is None:
            logging.warning("Either didn't receive joint states or error in FK calculation->pose_actual:None")
            return

        if self.null_space:
            if leftarm_f:
                q_dot_constraints=q_dot_constraints[0:7]
                theta = theta[0:7]
            elif rightarm_f:
                theta = theta[7:14]
                q_dot_constraints=q_dot_constraints[7:14]
            else:
                theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                q_dot_constraints=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            joint_states_dot_inst = JJT_inv @ (np.array(posedot).reshape(6,1)) #+ (np.eye(7)-JJT_inv@J)@ q_dot_constraints
        else:
            joint_states_dot_inst = JJT_inv @ (np.array(posedot).reshape(6,1))

        if self.joint_states_prev is None:
            print(f"Previous joints state is {self.joint_states_prev}, Please start hardware or simulation to get current joint states.")
            self.joint_states_prev = theta
        joint_states_desired = self.integrate_joint_positions(np.array(joint_states_dot_inst).flatten(), self.joint_states_prev, 1/self.sampling_frequency)
        self.joint_states_prev = joint_states_desired
        
        if t==0:
            print(f"-------------Executing trajectory-------------------")
            print(f"left:{leftarm_f}, right:{rightarm_f} arm final transformation:=> {T_final}")
        if leftarm_f and rightarm_f:
            joint_states_desired=np.array([0.0 for _ in range(7)])
        return joint_states_desired
        
    def get_joints(self, t, traj_time, theta, dh_l, T_init, T_final, leftarm_f=False, rightarm_f=False):
        return self.joints_trajectory(t, traj_time, theta, dh_l, T_init, T_final, leftarm_f, rightarm_f)
    
if __name__== "__main__":
    obj = TrajectoryGenerator()
    """_, T_init = obj.FK([-0.085, 1.396, -1.317, -1.365, 0.583, 0.221, -0.365], obj.dh_l)
    T_final = np.array([[0, -1, 0, 0], 
                                 [0, 0, -1, -.10555-0.176-0.3-0.32-0.2251], #0.10555,0.176,0.3,0.32,0.2251
                                 [1, 0, 0, 0], 
                                 [0, 0, 0, 1]])"""
    T_init = np.array([[0, -1, 0, 0], 
                                 [0, 0, -1, -0.5], 
                                 [1, 0, 0, 0], 
                                 [0, 0, 0, 1]])
    T_final = np.array([[0, 0, 1, 0.3], 
                                 [0, -1, 0, -0.19], 
                                 [1, 0, 0, -0.5], 
                                 [0, 0, 0, 1]])
    print(np.concatenate((obj.get_joints(9.0, 10, [0.1 for _ in range(14)], obj.dh_l, T_init, T_final,leftarm_f=True, rightarm_f=True), obj.get_joints(5.0, 10, [0.1 for _ in range(14)], obj.dh_l, T_init, T_final,leftarm_f=False, rightarm_f=True))))
   
