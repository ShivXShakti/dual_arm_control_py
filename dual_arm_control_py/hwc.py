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
        self.sampling_frequency = 100    
        self.timer = self.create_timer(1/self.sampling_frequency, self.timer_callback)  # 100 Hz
        self.obj = TrajectoryGenerator()
        self.t_counter = 0.0
        self.js_prev = None
        self.joint_state_actual = np.zeros((14,))

        #### hwc
        self.status = darm_msgs.msg.UiStatus()
        self.obj = TrajectoryGenerator()
        self.joint_callback_status = False
        self.pub = self.create_publisher(darm_msgs.msg.UiCommand,"svaya/ui/command",10)
        self.sub = self.create_subscription(darm_msgs.msg.UiStatus, "svaya/ui/status",self.callback,10)


    def timer_callback(self):
        if not self.joint_callback_status:
            self.get_logger().info(f"Did not receive joint states... trying again!")
            return
        else:
            if self.t_counter<self.obj.traj_time:
                angle = self.joint_state_actual
                for i in range(7):
                    angle[i] = self.status.left_arm.position[i]
                    angle[i+7] = self.status.right_arm.position[i]
                
                target_angle = self.obj.get_joints(self.t_counter, angle[7:14], self.obj.dh_l)
                self.js_prev = target_angle
                uicmd_msg  = darm_msgs.msg.UiCommand()
                uicmd_msg.developer_command.enable = True

                for i in range(16):
                    dmsg=  darm_msgs.msg.JointCommand()
                    if i<14:
                        dmsg.position = target_angle[i]
                    else:
                        dmsg.position = 0.0
                    dmsg.velocity = 0.0
                    uicmd_msg.developer_command.command.append(dmsg)
                self.t_counter += 1/self.sampling_frequency
                self.pub.publish(uicmd_msg)
            else:
                uicmd_msg  = darm_msgs.msg.UiCommand()
                uicmd_msg.developer_command.enable = True

                for i in range(16):
                    dmsg=  darm_msgs.msg.JointCommand()
                    if i<14:
                        dmsg.position = self.js_prev[i]
                    else:
                        dmsg.position = 0.0
                    dmsg.velocity = 0.0
                    uicmd_msg.developer_command.command.append(dmsg)
                self.get_logger().info(f"---------Execution completed.-------")
                self.pub.publish(uicmd_msg)

                
    def callback(self, msg):
        self.status = msg
        self.joint_state_actual = self.status.right_arm.position
        self.joint_callback_status  = True

def main(args=None):
    rclpy.init(args=args)
    node = PositionCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
