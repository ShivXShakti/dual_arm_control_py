import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
from dual_arm_control_py.online_trajectory_generator import TrajectoryGenerator
#from dual_arm_control_py.traj_generator_ik import TrajectoryGenerator
from sensor_msgs.msg import JointState
import numpy as np
import pandas as pd

class PositionCommander(Node):
    def __init__(self):
        super().__init__('position_commander')
        self.sampling_frequency = 100
        self.publisher_ = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        #self.publisher_ = self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
    
        self.timer = self.create_timer(1/self.sampling_frequency, self.timer_callback)  # 100 Hz
        self.start_time = self.get_clock().now().nanoseconds
        self.obj = TrajectoryGenerator()
        self.dh_l = self.obj.dh_l
        l1,l2,l3,l4,l5 = self.dh_l
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10
        )

        self.num_joints = 16  # Adjust according to your robot
        self.js = None
        self.joint_names = [
            'J1_left', 'J2_left', 'J3_left', 'J4_left', 'J5_left', 'J6_left', 'L_link7_to_flange',
            'J1_right', 'J2_right', 'J3_right', 'J4_right', 'J5_right', 'J6_right', 'R_link7_to_flange',
            'neck_joint', 'head_joint'
        ]
        self.ordered_positions = None #[0.0 for _ in range(len(self.joint_names))]
        self.t_counter = 0.0
        self.ct = 0
        self.obj_reach_l = 10
        self.obj_reach_r = 10

        self.T_init_l = np.array([[0, -1, 0, 0], 
                                 [0, 0, -1, 0.10555+0.176+0.3+0.32+0.225], #0.10555,0.176,0.3,0.32,0.2251
                                 [1, 0, 0, 0], 
                                 [0, 0, 0, 1]])
        
        self.T_final_l = np.array([[0, -1, 0, 0], 
                                 [1, 0, 0, l1+l2], #0.10555,0.176,0.3,0.32,0.2251
                                 [0, 0, 1, -l3-l4], 
                                 [0, 0, 0, 1]])
        
        self.T_init_r = np.array([[0, -1, 0, 0], 
                                 [0, 0, -1, -0.10555-0.176-0.3-0.32-0.225], #0.10555,0.176,0.3,0.32,0.2251
                                 [1, 0, 0, 0], 
                                 [0, 0, 0, 1]])
        self.T_final_r = np.array([[0, 0.866, 0.5, l4+l5], 
                                 [0, -0.5, 0.866, -0.3], #0.10555,0.176,0.3,0.32,0.2251
                                 [1, 0, 0, -l3], 
                                 [0, 0, 0, 1]])

    def listener_callback(self, msg):
        name_to_position = dict(zip(msg.name, msg.position))
        self.ordered_positions = [name_to_position.get(self.joint_names[i], float('nan')) for i in range(len(self.joint_names))]
    
    def timer_callback(self):
        if self.ordered_positions is None:
            return
        msg = Float64MultiArray()
        """joints = np.concatenate((self.obj.get_joints(t=self.t_counter, traj_time=self.obj_reach_l, theta=[self.ordered_positions[0], self.ordered_positions[1], self.ordered_positions[2], self.ordered_positions[3], self.ordered_positions[4], self.ordered_positions[5], self.ordered_positions[6], self.ordered_positions[7], self.ordered_positions[8], self.ordered_positions[9], self.ordered_positions[10], self.ordered_positions[11], self.ordered_positions[12], self.ordered_positions[13]], 
                                                     dh_l=self.obj.dh_l, T_init=self.T_init_l, T_final=self.T_final_l,leftarm_f=True, rightarm_f=False), 
                                 self.obj.get_joints(t=self.t_counter, traj_time=self.obj_reach_r, theta=[self.ordered_positions[0], self.ordered_positions[1], self.ordered_positions[2], self.ordered_positions[3], self.ordered_positions[4], self.ordered_positions[5], self.ordered_positions[6], self.ordered_positions[7], self.ordered_positions[8], self.ordered_positions[9], self.ordered_positions[10], self.ordered_positions[11], self.ordered_positions[12], self.ordered_positions[13]], dh_l=self.obj.dh_l, T_init=self.T_init_r, T_final=self.T_final_r,leftarm_f=False, rightarm_f=True)))"""
        joints = np.concatenate((self.obj.get_joints(t=self.t_counter, traj_time=self.obj_reach_l, theta=[self.ordered_positions[0], self.ordered_positions[1], self.ordered_positions[2], self.ordered_positions[3], self.ordered_positions[4], self.ordered_positions[5], self.ordered_positions[6], self.ordered_positions[7], self.ordered_positions[8], self.ordered_positions[9], self.ordered_positions[10], self.ordered_positions[11], self.ordered_positions[12], self.ordered_positions[13]], 
                                                     dh_l=self.obj.dh_l, T_init=self.T_init_l, T_final=self.T_final_l,leftarm_f=True, rightarm_f=False), 
                                 np.array([0.0 for _ in range(7)])))
        msg.data = [joints[0], joints[1], joints[2], joints[3], joints[4], joints[5], joints[6], joints[7], joints[8], joints[9], joints[10], joints[11], joints[12], joints[13], 0.0, 0.0]
        self.t_counter += 1/self.sampling_frequency
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PositionCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
