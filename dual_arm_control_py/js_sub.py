import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from dual_arm_control_py.traj_generator_ik import TrajectoryGenerator
from sensor_msgs.msg import JointState
import darm_msgs.msg
import numpy as np
import time
import threading

class PositionCommander(Node):
    def __init__(self):
        super().__init__('position_commander')
        self.joint_state_actual = np.zeros((14,))

        #### hwc
        self.status = darm_msgs.msg.UiStatus()
        self.joint_callback_status = False
        self.sub = self.create_subscription(darm_msgs.msg.UiStatus, "svaya/ui/status",self.callback,10)

                
    def callback(self, msg):
        self.status = msg
        self.joint_state_actual = self.status.right_arm.position
        self.joint_callback_status  = True
        self.get_logger().info(f"Joint states actual:{self.joint_state_actual}, js_status: {self.joint_callback_status}")

def main(args=None):
    rclpy.init(args=args)
    node = PositionCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
