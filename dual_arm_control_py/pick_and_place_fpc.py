import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import math
#from dual_arm_control_py.traj_generator_ik_with_nullspaceconstraints import TrajectoryGenerator
from dual_arm_control_py.online_joints_trajectory_generator import TrajectoryGenerator

#from dual_arm_control_py.traj_generator_ik import TrajectoryGenerator
from sensor_msgs.msg import JointState
import numpy as np
import pandas as pd
import time

class PositionCommander(Node):
    def __init__(self):
        super().__init__('position_commander')
        self.sampling_frequency = 100
        self.publisher_ = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)    
        self.gripper_publisher_ = self.create_publisher(JointTrajectory, '/gripper_position_controller/joint_trajectory', 10)
        self.timer = self.create_timer(1/self.sampling_frequency, self.timer_callback)  # 100 Hz
        self.start_time = self.get_clock().now().nanoseconds
        self.obj = TrajectoryGenerator()
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
        self.obj_reach_t = 10
        self.obj_put_t = 10
        self.return_t = 10
        self.grasp_t = 2
        self.grasp_f = False
        self.traj_execution_f = False
        l1,l2, l3, l4, l5 = [0.10555,0.176,0.3,0.32,0.2251]
        self.T_init = np.array([[0, -1, 0, 0], 
                                 [0, 0, -1, -l1-l2-l3-l4-l5], 
                                 [1, 0, 0, 0], 
                                 [0, 0, 0, 1]])
        self.T_obj = np.array([[0, 0, 1, l4+l5], 
                                 [0, -1, 0, -0.19], 
                                 [1, 0, 0, -0.5], 
                                 [0, 0, 0, 1]])
        self.T_rot = np.array([[1, 0, 0, 0], 
                                 [0, 0, -1, -l1-l2-l3-l4-l5], 
                                 [0, 1, 0, 0], 
                                 [0, 0, 0, 1]])

    def listener_callback(self, msg):
        name_to_position = dict(zip(msg.name, msg.position))
        self.ordered_positions = [name_to_position.get(self.joint_names[i], float('nan')) for i in range(len(self.joint_names))]
    
    def timer_callback(self):
        if self.ordered_positions is None:
            return
        
        else:
            if self.t_counter < self.obj_reach_t:
                msg = Float64MultiArray()
                joints = self.obj.get_joints(t=self.t_counter, traj_time=self.obj_reach_t, theta=[self.ordered_positions[7], self.ordered_positions[8], self.ordered_positions[9], self.ordered_positions[10], self.ordered_positions[11], self.ordered_positions[12], self.ordered_positions[13]], dh_l=self.obj.dh_l, T_init=self.T_init, T_final=self.T_obj)
                msg.data = [joints[0], joints[1], joints[2], joints[3], joints[4], joints[5], joints[6], joints[7], joints[8], joints[9], joints[10], joints[11], joints[12], joints[13], 0.0, 0.0]
                self.t_counter += 1/self.sampling_frequency
                self.publisher_.publish(msg)
            elif self.t_counter <= self.obj_reach_t + self.grasp_t:
                if not self.grasp_f:
                    traj = JointTrajectory()
                    traj.joint_names = ["L_F1M1", "L_F1M2", "L_F1M3", "L_F1M4", "L_F2M1", "L_F2M2", "L_F2M3", "L_F2M4", "L_F3M1", "L_F3M2", "L_F3M3", "L_F3M4", 
                            "R_F1M1", "R_F1M2", "R_F1M3", "R_F1M4", "R_F2M1", "R_F2M2", "R_F2M3", "R_F2M4", "R_F3M1", "R_F3M2", "R_F3M3", "R_F3M4"]  # Replace with your actual joint name
                    point = JointTrajectoryPoint()
                    point.positions = [0.9]*24
                    point.time_from_start.sec = 1
                    traj.points.append(point)
                    self.gripper_publisher_.publish(traj)
                    self.get_logger().info('Sent gripper position command')
                    time.sleep(self.grasp_t)
                    self.grasp_f = True
                self.t_counter += 1/self.sampling_frequency
            elif self.t_counter >= (self.obj_reach_t+self.grasp_t) and self.t_counter < (self.obj_reach_t+self.grasp_t+self.obj_put_t):
                msg = Float64MultiArray()
                joints = self.obj.get_joints(t=self.t_counter-(self.obj_reach_t+self.grasp_t), traj_time=self.obj_reach_t+self.grasp_t, theta=[self.ordered_positions[7], self.ordered_positions[8], self.ordered_positions[9], self.ordered_positions[10], self.ordered_positions[11], self.ordered_positions[12], self.ordered_positions[13]], dh_l=self.obj.dh_l, T_init=self.T_obj, T_final=self.T_rot)
                msg.data = [joints[0], joints[1], joints[2], joints[3], joints[4], joints[5], joints[6], joints[7], joints[8], joints[9], joints[10], joints[11], joints[12], joints[13], 0.0, 0.0]
                self.t_counter += 1/self.sampling_frequency
                self.publisher_.publish(msg)
            elif not self.traj_execution_f and self.t_counter >= (self.obj_reach_t+self.grasp_t +self.obj_put_t) and self.t_counter <= (self.obj_reach_t+self.grasp_t+self.obj_put_t+self.return_t):
                msg = Float64MultiArray()
                joints = self.obj.get_joints(t=self.t_counter-(self.obj_reach_t+self.grasp_t +self.obj_put_t), traj_time=self.obj_reach_t+self.grasp_t +self.obj_put_t ,theta=[self.ordered_positions[7], self.ordered_positions[8], self.ordered_positions[9], self.ordered_positions[10], self.ordered_positions[11], self.ordered_positions[12], self.ordered_positions[13]], dh_l=self.obj.dh_l, T_init=self.T_rot, T_final= self.T_init)
                msg.data = [joints[0], joints[1], joints[2], joints[3], joints[4], joints[5], joints[6], joints[7], joints[8], joints[9], joints[10], joints[11], joints[12], joints[13], 0.0, 0.0]
                self.t_counter += 1/self.sampling_frequency
                self.publisher_.publish(msg)
                if self.t_counter>=self.obj_reach_t+self.grasp_t+self.obj_put_t+self.return_t:
                    self.get_logger().info(f"----------Trajectory executed.----------")
                    self.traj_execution_f = True

            
def main(args=None):
    rclpy.init(args=args)
    node = PositionCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
