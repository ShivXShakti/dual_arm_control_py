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
    def __init__(self, sampling_frequency=100, dh_l=[0.10555,0.176,0.3,0.32,0.2251]):
        self.traj_status = True
        self.dh_l = dh_l
        self.sampling_frequency = sampling_frequency
        #------- trajectory execution flags -------
        self.grip_reach_f = False
        self.put_reach_f = False
        #-------- null space constraints----------
        self.max_joints_limit = np.array([3.14,1.56,3.141,2.446,3.141,1.306,6.28,3.14,1.56,3.141,2.446,3.141,1.306,6.28])
        self.min_joints_limit = np.array([-3.141,-1.56,-3.141,-1.365,-3.141,-1.306,-6.28,-3.141,-1.56,-3.141,-1.365,-3.141,-1.306,-6.28])
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
        self.joint_states_prev = None #[-0.085, 1.396, -1.317, -1.365, 0.583, 0.221, -0.365, -0.085, 1.396, -1.317, -1.365, 0.583, 0.221, -0.365]#[3.14,-1.5699755717780695, 1.57,-1.569931101434043, -1.57, 1.57]
        #self.joint_angles_both = None
    
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
    
    def compute_jacobian(self, theta, dh_l):
        theta = np.array(theta[0])
        print(theta)
        print(theta.shape)
        l1,l2,l3,l4,l5 = dh_l[0], dh_l[1], dh_l[2], dh_l[3], dh_l[4]
        theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, theta10, theta11, theta12, theta13, theta14 = theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7], theta[8], theta[9], theta[10], theta[11], theta[12], theta[13]

        J_l = np.array([[0, -np.cos(theta1), -np.sin(theta1)*np.sin(theta2), np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3), 
               (-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.sin(theta4) - np.sin(theta1)*np.sin(theta2)*np.cos(theta4), 
               -((-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.cos(theta4) + np.sin(theta1)*np.sin(theta2)*np.sin(theta4))*np.sin(theta5) + (np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3))*np.cos(theta5), 
               (((-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.cos(theta4) + np.sin(theta1)*np.sin(theta2)*np.sin(theta4))*np.cos(theta5) + (np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3))*np.sin(theta5))*np.sin(theta6) + ((-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.sin(theta4) - np.sin(theta1)*np.sin(theta2)*np.cos(theta4))*np.cos(theta6)],
     
            [-1, 0, -np.cos(theta2), -np.sin(theta2)*np.sin(theta3), np.sin(theta2)*np.sin(theta4)*np.cos(theta3) - np.cos(theta2)*np.cos(theta4), 
             -(np.sin(theta2)*np.cos(theta3)*np.cos(theta4) + np.sin(theta4)*np.cos(theta2))*np.sin(theta5) - np.sin(theta2)*np.sin(theta3)*np.cos(theta5), 
             ((np.sin(theta2)*np.cos(theta3)*np.cos(theta4) + np.sin(theta4)*np.cos(theta2))*np.cos(theta5) - np.sin(theta2)*np.sin(theta3)*np.sin(theta5))*np.sin(theta6) + (np.sin(theta2)*np.sin(theta4)*np.cos(theta3) - np.cos(theta2)*np.cos(theta4))*np.cos(theta6)],
            
            [0, -np.sin(theta1), np.sin(theta2)*np.cos(theta1), -np.sin(theta1)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1)*np.cos(theta2), (-np.sin(theta1)*np.sin(theta3) + np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.sin(theta4) + np.sin(theta2)*np.cos(theta1)*np.cos(theta4), 
             -((-np.sin(theta1)*np.sin(theta3) + np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.cos(theta4) - np.sin(theta2)*np.sin(theta4)*np.cos(theta1))*np.sin(theta5) + (-np.sin(theta1)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.cos(theta5), 
             (((-np.sin(theta1)*np.sin(theta3) + np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.cos(theta4) - np.sin(theta2)*np.sin(theta4)*np.cos(theta1))*np.cos(theta5) + (-np.sin(theta1)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.sin(theta5))*np.sin(theta6) + ((-np.sin(theta1)*np.sin(theta3) + np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.sin(theta4) + np.sin(theta2)*np.cos(theta1)*np.cos(theta4))*np.cos(theta6)],
     
            [(-0.2251*((np.sin(theta1)*np.sin(theta3) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.cos(theta4) + np.sin(theta2)*np.sin(theta4)*np.cos(theta1))*np.cos(theta5) - 0.2251*(np.sin(theta1)*np.cos(theta3) + np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.sin(theta5))*np.sin(theta6) + (-0.2251*(np.sin(theta1)*np.sin(theta3) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.sin(theta4) + 0.2251*np.sin(theta2)*np.cos(theta1)*np.cos(theta4))*np.cos(theta6) + (-0.32*np.sin(theta1)*np.sin(theta3) + 0.32*np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.sin(theta4) + 0.32*np.sin(theta2)*np.cos(theta1)*np.cos(theta4) + 0.3*np.sin(theta2)*np.cos(theta1), 
             (-0.2251*(np.sin(theta1)*np.sin(theta2)*np.cos(theta3)*np.cos(theta4) + np.sin(theta1)*np.sin(theta4)*np.cos(theta2))*np.cos(theta5) + 0.2251*np.sin(theta1)*np.sin(theta2)*np.sin(theta3)*np.sin(theta5))*np.sin(theta6) + (-0.2251*np.sin(theta1)*np.sin(theta2)*np.sin(theta4)*np.cos(theta3) + 0.2251*np.sin(theta1)*np.cos(theta2)*np.cos(theta4))*np.cos(theta6) - 0.32*np.sin(theta1)*np.sin(theta2)*np.sin(theta4)*np.cos(theta3) + 0.32*np.sin(theta1)*np.cos(theta2)*np.cos(theta4) + 0.3*np.sin(theta1)*np.cos(theta2), 
             (-0.2251*(np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3))*np.cos(theta4)*np.cos(theta5) - 0.2251*(np.sin(theta1)*np.cos(theta2)*np.cos(theta3) + np.sin(theta3)*np.cos(theta1))*np.sin(theta5))*np.sin(theta6) + (-0.32*np.sin(theta1)*np.sin(theta3)*np.cos(theta2) + 0.32*np.cos(theta1)*np.cos(theta3))*np.sin(theta4) - 0.2251*(np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3))*np.sin(theta4)*np.cos(theta6), 
             -0.2251*(-(-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.sin(theta4) + np.sin(theta1)*np.sin(theta2)*np.cos(theta4))*np.sin(theta6)*np.cos(theta5) + (-0.2251*(-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.cos(theta4) - 0.2251*np.sin(theta1)*np.sin(theta2)*np.sin(theta4))*np.cos(theta6) + (0.32*np.sin(theta1)*np.cos(theta2)*np.cos(theta3) + 0.32*np.sin(theta3)*np.cos(theta1))*np.cos(theta4) - 0.32*np.sin(theta1)*np.sin(theta2)*np.sin(theta4), 
             (0.2251*((-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.cos(theta4) + np.sin(theta1)*np.sin(theta2)*np.sin(theta4))*np.sin(theta5) - 0.2251*(np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3))*np.cos(theta5))*np.sin(theta6), 
             (-0.2251*((-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.cos(theta4) + np.sin(theta1)*np.sin(theta2)*np.sin(theta4))*np.cos(theta5) - 0.2251*(np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3))*np.sin(theta5))*np.cos(theta6) - (-0.2251*(-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.sin(theta4) + 0.2251*np.sin(theta1)*np.sin(theta2)*np.cos(theta4))*np.sin(theta6), 0], 
            
            [0, (-0.2251*(-np.sin(theta2)*np.sin(theta4) + np.cos(theta2)*np.cos(theta3)*np.cos(theta4))*np.cos(theta5) + 0.2251*np.sin(theta3)*np.sin(theta5)*np.cos(theta2))*np.sin(theta6) + (-0.2251*np.sin(theta2)*np.cos(theta4) - 0.2251*np.sin(theta4)*np.cos(theta2)*np.cos(theta3))*np.cos(theta6) - 0.32*np.sin(theta2)*np.cos(theta4) - 0.3*np.sin(theta2) - 0.32*np.sin(theta4)*np.cos(theta2)*np.cos(theta3), (0.2251*np.sin(theta2)*np.sin(theta3)*np.cos(theta4)*np.cos(theta5) + 0.2251*np.sin(theta2)*np.sin(theta5)*np.cos(theta3))*np.sin(theta6) + 0.2251*np.sin(theta2)*np.sin(theta3)*np.sin(theta4)*np.cos(theta6) + 0.32*np.sin(theta2)*np.sin(theta3)*np.sin(theta4), 
             -0.2251*(-np.sin(theta2)*np.sin(theta4)*np.cos(theta3) + np.cos(theta2)*np.cos(theta4))*np.sin(theta6)*np.cos(theta5) + (-0.2251*np.sin(theta2)*np.cos(theta3)*np.cos(theta4) - 0.2251*np.sin(theta4)*np.cos(theta2))*np.cos(theta6) - 0.32*np.sin(theta2)*np.cos(theta3)*np.cos(theta4) - 0.32*np.sin(theta4)*np.cos(theta2), 
             (0.2251*(np.sin(theta2)*np.cos(theta3)*np.cos(theta4) + np.sin(theta4)*np.cos(theta2))*np.sin(theta5) + 0.2251*np.sin(theta2)*np.sin(theta3)*np.cos(theta5))*np.sin(theta6), (-0.2251*(np.sin(theta2)*np.cos(theta3)*np.cos(theta4) + np.sin(theta4)*np.cos(theta2))*np.cos(theta5) + 0.2251*np.sin(theta2)*np.sin(theta3)*np.sin(theta5))*np.cos(theta6) - (-0.2251*np.sin(theta2)*np.sin(theta4)*np.cos(theta3) + 0.2251*np.cos(theta2)*np.cos(theta4))*np.sin(theta6), 0],
     
            [(-0.2251*((-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.cos(theta4) + np.sin(theta1)*np.sin(theta2)*np.sin(theta4))*np.cos(theta5) - 0.2251*(np.sin(theta1)*np.sin(theta3)*np.cos(theta2) - np.cos(theta1)*np.cos(theta3))*np.sin(theta5))*np.sin(theta6) + (-0.2251*(-np.sin(theta1)*np.cos(theta2)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1))*np.sin(theta4) + 0.2251*np.sin(theta1)*np.sin(theta2)*np.cos(theta4))*np.cos(theta6) + (0.32*np.sin(theta1)*np.cos(theta2)*np.cos(theta3) + 0.32*np.sin(theta3)*np.cos(theta1))*np.sin(theta4) + 0.32*np.sin(theta1)*np.sin(theta2)*np.cos(theta4) + 0.3*np.sin(theta1)*np.sin(theta2), 
             (-0.2251*(-np.sin(theta2)*np.cos(theta1)*np.cos(theta3)*np.cos(theta4) - np.sin(theta4)*np.cos(theta1)*np.cos(theta2))*np.cos(theta5) - 0.2251*np.sin(theta2)*np.sin(theta3)*np.sin(theta5)*np.cos(theta1))*np.sin(theta6) + (0.2251*np.sin(theta2)*np.sin(theta4)*np.cos(theta1)*np.cos(theta3) - 0.2251*np.cos(theta1)*np.cos(theta2)*np.cos(theta4))*np.cos(theta6) + 0.32*np.sin(theta2)*np.sin(theta4)*np.cos(theta1)*np.cos(theta3) - 0.32*np.cos(theta1)*np.cos(theta2)*np.cos(theta4) - 0.3*np.cos(theta1)*np.cos(theta2), 
             (-0.2251*(np.sin(theta1)*np.sin(theta3) - np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.sin(theta5) - 0.2251*(-np.sin(theta1)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.cos(theta4)*np.cos(theta5))*np.sin(theta6) - 0.2251*(-np.sin(theta1)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.sin(theta4)*np.cos(theta6) + (0.32*np.sin(theta1)*np.cos(theta3) + 0.32*np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.sin(theta4), 
             -0.2251*(-(-np.sin(theta1)*np.sin(theta3) + np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.sin(theta4) - np.sin(theta2)*np.cos(theta1)*np.cos(theta4))*np.sin(theta6)*np.cos(theta5) + (-0.2251*(-np.sin(theta1)*np.sin(theta3) + np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.cos(theta4) + 0.2251*np.sin(theta2)*np.sin(theta4)*np.cos(theta1))*np.cos(theta6) + (0.32*np.sin(theta1)*np.sin(theta3) - 0.32*np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.cos(theta4) + 0.32*np.sin(theta2)*np.sin(theta4)*np.cos(theta1), 
             (0.2251*((-np.sin(theta1)*np.sin(theta3) + np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.cos(theta4) - np.sin(theta2)*np.sin(theta4)*np.cos(theta1))*np.sin(theta5) - 0.2251*(-np.sin(theta1)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.cos(theta5))*np.sin(theta6), 
             (-0.2251*((-np.sin(theta1)*np.sin(theta3) + np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.cos(theta4) - np.sin(theta2)*np.sin(theta4)*np.cos(theta1))*np.cos(theta5) - 0.2251*(-np.sin(theta1)*np.cos(theta3) - np.sin(theta3)*np.cos(theta1)*np.cos(theta2))*np.sin(theta5))*np.cos(theta6) - (-0.2251*(-np.sin(theta1)*np.sin(theta3) + np.cos(theta1)*np.cos(theta2)*np.cos(theta3))*np.sin(theta4) - 0.2251*np.sin(theta2)*np.cos(theta1)*np.cos(theta4))*np.sin(theta6), 0]])

        J_r = np.array([[0, np.cos(theta8), np.sin(theta8)*np.sin(theta9), -np.sin(theta8)*np.sin(theta10)*np.cos(theta9) + np.cos(theta8)*np.cos(theta10), -(-np.sin(theta8)*np.cos(theta9)*np.cos(theta10) - np.sin(theta10)*np.cos(theta8))*np.sin(theta11) + np.sin(theta8)*np.sin(theta9)*np.cos(theta11), 
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
        return J_l, J_r
    
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
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14 = np.array(theta[0])
        l1,l2, l3, l4, l5 = dh_l[0], dh_l[1], dh_l[2], dh_l[3], dh_l[4] #np.array([0.10555,0.176,0.3,0.32,0.2251])
        dh_right = np.array([
            [0,        np.pi/2,      l1,      t1+np.pi/2],
            [0,        0,            l2,      0],
            [0,        np.pi/2,      0,      t2],
            [0,        -np.pi/2,     l3,      t3],
            [0,        np.pi/2,      0,      t4],
            [0,        -np.pi/2,     l4,      t5],
            [0,        np.pi/2,      0,      t6],
            [0,        -np.pi/2,     l5,      t7]
        ])

        dh_left = np.array([
            [0.0, np.pi/2,       -l1, t8+np.pi/2],
            [0.0, 0.0,         -l2, 0.0],
            [0.0, -np.pi/2,       0.0, t9],
            [0.0, np.pi/2,   -l3, t10],
            [0.0, -np.pi/2,   0.0, t11],
            [0.0,  np.pi/2,   -l4, t12],
            [0.0,   -np.pi/2,      0.0, t13],
            [0.0,   np.pi/2,      -l5, t14]
        ])
    
        # Homogeneous Transformation Matrices
        T01l = self.t_matrix(dh_left[0,0], dh_left[0,1], dh_left[0,2], dh_left[0,3])
        T1dl = self.t_matrix(dh_left[1,0], dh_left[1,1], dh_left[1,2], dh_left[1,3])
        Td2l = self.t_matrix(dh_left[2,0], dh_left[2,1], dh_left[2,2], dh_left[2,3])
        T23l = self.t_matrix(dh_left[3,0], dh_left[3,1], dh_left[3,2], dh_left[3,3])
        T34l = self.t_matrix(dh_left[4,0], dh_left[4,1], dh_left[4,2], dh_left[4,3])
        T45l = self.t_matrix(dh_left[5,0], dh_left[5,1], dh_left[5,2], dh_left[5,3])
        T56l = self.t_matrix(dh_left[6,0], dh_left[6,1], dh_left[6,2], dh_left[6,3])
        T67l = self.t_matrix(dh_left[7,0], dh_left[7,1], dh_left[7,2], dh_left[7,3])
        Tl = T01l @T1dl @ Td2l @ T23l @ T34l @ T45l @ T56l @T67l
        angles_l = self.rot2eul(Tl[:3,:3])
        position_l = Tl[:3,3]


        T01r = self.t_matrix(dh_right[0,0], dh_right[0,1], dh_right[0,2], dh_right[0,3])
        T1dr = self.t_matrix(dh_right[1,0], dh_right[1,1], dh_right[1,2], dh_right[1,3])
        Td2r = self.t_matrix(dh_right[2,0], dh_right[2,1], dh_right[2,2], dh_right[2,3])
        T23r = self.t_matrix(dh_right[3,0], dh_right[3,1], dh_right[3,2], dh_right[3,3])
        T34r = self.t_matrix(dh_right[4,0], dh_right[4,1], dh_right[4,2], dh_right[4,3])
        T45r = self.t_matrix(dh_right[5,0], dh_right[5,1], dh_right[5,2], dh_right[5,3])
        T56r = self.t_matrix(dh_right[6,0], dh_right[6,1], dh_right[6,2], dh_right[6,3])
        T67r = self.t_matrix(dh_right[7,0], dh_right[7,1], dh_right[7,2], dh_right[7,3])
        Tr = T01r @T1dr @ Td2r @ T23r @ T34r @ T45r @ T56r @T67r
        angles_r = self.rot2eul(Tr[:3,:3])
        position_r = Tr[:3,3]
        return np.concatenate((angles_l, position_l)), np.concatenate((angles_r, position_r)), Tl, Tr


    def joints_trajectory(self, t, traj_time_l, theta, dh_l, T_init, T_final):
        """ Generate smooth trajectory using the exponential matrix method. """
        if t>traj_time_l:
            t = traj_time_l
            if self.traj_status:
                print(f"---------------left hand trajectory execution complete.-----------------")
                self.traj_status = False
            return self.joint_angles_both

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
        #### ik
        J_left, J_right = self.compute_jacobian(theta, dh_l) # 6x7
        JTJ_left = J_left.T @ J_left     ## 7x7
        JTJ_left_ = JTJ_left + 0.001*np.eye(7)

        JTJ_right = J_right.T @ J_right     ## 7x7
        JTJ_right_ = JTJ_right + 0.001*np.eye(7)

        if np.linalg.det(JTJ_left_) != 0:
            JTJ_inv_left = np.linalg.inv(JTJ_left_) @ J_left.T
            JTJ_inv_right = np.linalg.inv(JTJ_right_) @ J_right.T
        else:
            JTJ_inv_left = np.linalg.pinv(JTJ_left_) @ J_left.T
            JTJ_inv_right = np.linalg.pinv(JTJ_right_) @ J_right.T 

        pose_al, pose_ar, _, _ = self.FK(theta, dh_l)   # [position, orientation]
        if pose_al is None:
            logging.warning("Either didn't receive joint states or error in FK calculation->pose_actual:None")
            return
        
        #### Null space condition
        q_dot_constraints = (self.mean_joints_limit.reshape(14,1)-np.array(theta).reshape(14,1))/self.max_min_dif_sq.reshape(14,1)
        if self.null_space:
            #joint_states_dot_inst = JJT_inv @ ((np.array(posedot).reshape(6,1)) + self.k_ @ (np.array(pose).reshape(6,1)-np.array(pose_a).reshape(6,1))) + (np.eye(7)-JJT_inv@J)@ q_dot_constraints
            joint_states_dot_inst = np.concatenate((JTJ_inv_left @ (np.array(posedot).reshape(6,1)) + (np.eye(7)-JTJ_inv_left@J_left)@ q_dot_constraints[0:7], JTJ_inv_right @ (np.array(posedot).reshape(6,1)) + (np.eye(7)-JTJ_inv_right@J_right)@ q_dot_constraints[7:14]))
        else:
            joint_states_dot_inst = np.concatenate((JTJ_inv_left @ (np.array(posedot).reshape(6,1)), JTJ_inv_right @ (np.array(posedot).reshape(6,1))))
        
        if self.joint_states_prev is None:
            print("previous pose is none")
            self.joint_states_prev = theta
        joint_states_desired = self.integrate_joint_positions(np.array(joint_states_dot_inst).flatten(), self.joint_states_prev, 1/self.sampling_frequency)
        #joint_angles_both = np.concatenate((joint_states_desired, joint_states_desired))
        self.joint_states_prev = joint_states_desired
        #self.joint_angles_both = joint_states_desired
        if t==0:
            print(f"-------------Executing trajectory-------------------")
            print(f"Final Transformation: {T_final}")
        return joint_states_desired[0]
        
    def get_joints(self, t, traj_time, theta, dh_l, T_init, T_final):
        return self.joints_trajectory(t, traj_time, theta, dh_l, T_init, T_final)
    
if __name__== "__main__":
    obj = TrajectoryGenerator()
    _, T_init = obj.FK([-0.085, 1.396, -1.317, -1.365, 0.583, 0.221, -0.365, -0.085, 1.396, -1.317, -1.365, 0.583, 0.221, -0.365], obj.dh_l)
    T_final = np.array([[0, -1, 0, 0], 
                                 [0, 0, -1, -.10555-0.176-0.3-0.32-0.2251], #0.10555,0.176,0.3,0.32,0.2251
                                 [1, 0, 0, 0], 
                                 [0, 0, 0, 1]])
    print(obj.get_joints(t=0.09, traj_time=20,theta=[0.1 for _ in range(14)], dh_l=obj.dh_l, T_init=T_init, T_final=T_final))
   
