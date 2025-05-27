from darm_api.darm_api import DarmAPI
import darm_msgs.msg
import rclpy
import numpy as np
import time
import threading
# from python_pkg.python_pkg import traj_generator_ik
from python_pkg.traj_generator_ik import TrajectoryGenerator
import rclpy.node
from rclpy.node import Node
import rclpy.time
import sensor_msgs.msg

class taskplanner(Node):
    def __init__(self):
        super().__init__('test3node')
        self.status = darm_msgs.msg.UiStatus()
        self.obj = TrajectoryGenerator()
        self.called = False
        self.pub = self.create_publisher(darm_msgs.msg.UiCommand,"svaya/ui/command",10)
        self.sub = self.create_subscription(darm_msgs.msg.UiStatus, "svaya/ui/status",self.callback,10)
        self.joint_state_pub = self.create_publisher(sensor_msgs.msg.JointState,"/svaya/joint_states",10)
        t1 = threading.Thread(target= self.run)
        t1.start()
        self.joint_state_msg = sensor_msgs.msg.JointState()
        self.joint_state_msg.position = [0.0]*30
        self.joint_state_msg.name = ["J1_left",
            "J2_left",
            "J3_left",
            "J4_left",
            "J5_left",
            "J6_left",
            "J7_left",
            "J1_right",
            "J2_right",
            "J3_right",
            "J4_right",
            "J5_right",
            "J6_right",
            "J7_right",
            "neck_joint",
            "head_joint",
            "L_F1M1",
            "L_F1M2",
            "L_F1M3",
            "L_F1M4",
            "L_F2M1",
            "L_F2M2",
            "L_F2M3",
            "L_F2M4",
            "L_F3M1",
            "L_F3M2",
            "L_F3M3",
            "L_F3M4",
            "L_F4M1",
            "L_F4M2",
            "L_F4M3",
            "L_F4M4",
            "R_F1M1",
            "R_F1M2",
            "R_F1M3",
            "R_F1M4",
            "R_F2M1",
            "R_F2M2",
            "R_F2M3",
            "R_F2M4",
            "R_F3M1",
            "R_F3M2",
            "R_F3M3",
            "R_F3M4",
            "R_F4M1",
            "R_F4M2",
            "R_F4M3",
            "R_F4M4"]



        # t1.join()
    def callback(self, msg):
        self.status = msg
        if not self.called:
            self.obj.joint_states = self.status.right_arm.position
            self.obj.joint_states_prev = self.status.right_arm.position
        else:
            self.obj.joint_states = self.obj.joint_states_prev
        self.called  = True
        # print(self.status.right_arm.position)

    def run(self):
        while(not self.called):
            # print("BIHIHI")
            time.sleep(.1)
        t_start = time.time()
        while(rclpy.ok()):
            t = time.time() - t_start
            # print(t)
            if t<self.obj.traj_time:
                angle = np.zeros((14,))
                # print(self.status.left_arm.position)
                for i in range(7):
                    angle[i] = self.status.left_arm.position[i]
                    angle[i+7] = self.status.right_arm.position[i]
                target_angle = self.obj.get_joints(t,angle[7:14])
                # print("Command publishing...")
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
                self.pub.publish(uicmd_msg)
                i=0
                for i in range(30):
                    # print(i)
                    if i<14:
                        self.joint_state_msg.position[i] = target_angle[i]
                    else:
                        self.joint_state_msg.position[i] = 0.0
                self.joint_state_msg.header.stamp = self.get_clock().now().to_msg()
                # self.joint_state_msg.header.stamp.nanosec = unsigned_int(time.time()*1000000000)
                self.joint_state_pub.publish(self.joint_state_msg)
                time.sleep(0.01)
            # print("Command Published")



def main(args=None):
    rclpy.init(args=args)
    try:
        obj = taskplanner()
        rclpy.spin(obj)
            
    except Exception as e:
        print(f"Error in the node: {e}")
    rclpy.shutdown()

if __name__ == '__main__':
    main()
