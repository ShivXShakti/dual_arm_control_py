import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
from dual_arm_control_py.home_traj_generator import TrajectoryGenerator
#from dual_arm_control_py.traj_generator_ik import TrajectoryGenerator
from sensor_msgs.msg import JointState
import numpy as np
import pandas as pd
import time

class PositionCommander(Node):
    def __init__(self):
        super().__init__('position_commander')
        self.sampling_frequency = 3
        self.publisher_ = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        #self.publisher_ = self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
    
        self.timer = self.create_timer(1/self.sampling_frequency, self.timer_callback)  # 100 Hz

        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10
        )

        self.t_counter = 0.0
        self.ct = 0
        self.js_callback = False
        self.gripper_pose_f = False

    def listener_callback(self, msg):
        name_to_position = dict(zip(msg.name, msg.position))
    
    def timer_callback(self):
        if self.t_counter<3:
            print(f"less than 3")
            
        elif not self.gripper_pose_f:
            self.gripper_pose_f = True
            print(f"goint to sleep....")
            time.sleep(10)
            print(f"slept for 3 sec.")
            
        else:
            print(f"else cond.")
        print(f"outside if else.")
        
        self.t_counter += 1/self.sampling_frequency

def main(args=None):
    rclpy.init(args=args)
    node = PositionCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
