#!/usr/bin/env python3
import darm_msgs.msg
import gripper_msgs.msg
import fts_msgs.msg
import rclpy
from rclpy.node import Node
import numpy as np
import time 


class GripperControl(Node):
    def __init__(self):
        super().__init__('api_node')
        self.left_joint_status_ = np.zeros((7,0))
        self.right_joint_status_ = np.zeros((7,0))
        self.head_joint_status_ = np.zeros((2,0))
        self.la_in_exe_ = False
        self.ra_in_exe_ = False 
        self.h_in_exe_ = False
        self.stop_robot_ = False
        self.gripper_status_ = gripper_msgs.msg.TesolloStatus()
        self.fts_status_ = fts_msgs.msg.FtsData()
        self.gripper_close_f = False
        

        self.gripper_cmd_pub_ = self.create_publisher(gripper_msgs.msg.TesolloCommand,"/svaya/ui/gripper/command",10)
        self.gripper_status_sub_ = self.create_subscription(gripper_msgs.msg.TesolloStatus, "/svaya/ui/gripper/status",self.GripperStatusCallback,10)
        self.timer = self.create_timer(0.01, self.timer_callback) 
        self.gripper_callback_status = False
        self.right_gripper_go(gripper_motor_angle=np.zeros((12,)),force = np.zeros((12,)),speed = np.zeros((12,)))
   

    def GripperStatusCallback(self,msg):
        self.gripper_status_ = msg
        self.gripper_callback_status = True

    
    def left_gripper_go(self,gripper_motor_angle,force = np.zeros((12,)),speed = np.zeros((12,))):
        if len(gripper_motor_angle) != 12:
            print("wrong gripper joint entry: the current size is: ", len(gripper_motor_angle))
            return
        grip_msg_list = []
        gripper_cmd_msg = gripper_msgs.msg.TesolloCommand()
        gripper_cmd_msg.part.id = gripper_msgs.msg.GripperPart.LEFT_GRIPPER
        for gripper in range(0,2):
            grip_msg = gripper_msgs.msg.TesolloCommandData()
            for finger in range(3):
                fing_msg = gripper_msgs.msg.TesolloFingerCommand()
                pos = []; vel=[]; grip_force = []
                for motor in range(4):
                    pos.append(float(gripper_motor_angle[finger*4 + motor]))
                    vel.append(float(speed[finger*4 + motor]))
                    grip_force.append(float(force[finger*4 + motor]))
                fing_msg.position = pos 
                fing_msg.force = grip_force 
                fing_msg.speed = vel
                grip_msg.finger_command.append(fing_msg)
            grip_msg.grasp_mode.mode = gripper_msgs.msg.TesolloGraspingMode.BASIC_MODE
            grip_msg.individual_control = 1
            grip_msg_list.append(grip_msg)
        gripper_cmd_msg.command = grip_msg_list
        self.gripper_cmd_pub_.publish(gripper_cmd_msg)
        
    def right_gripper_go(self,gripper_motor_angle,force = np.zeros((12,)),speed = np.zeros((12,))):
        if len(gripper_motor_angle) != 12:
            print("wrong gripper joint entry: the current size is: ", len(gripper_motor_angle))
            return
        grip_msg_list = []
        gripper_cmd_msg = gripper_msgs.msg.TesolloCommand()
        gripper_cmd_msg.part.id = gripper_msgs.msg.GripperPart.RIGHT_GRIPPER
        for gripper in range(0,2):
            grip_msg = gripper_msgs.msg.TesolloCommandData()
            for finger in range(3):
                fing_msg = gripper_msgs.msg.TesolloFingerCommand()
                pos = []; vel=[]; grip_force = []
                for motor in range(4):
                    pos.append(float(gripper_motor_angle[finger*4 + motor]))
                    vel.append(float(speed[finger*4 + motor]))
                    grip_force.append(float(force[finger*4 + motor]))
                fing_msg.position = pos 
                fing_msg.force = grip_force 
                fing_msg.speed = vel
                grip_msg.finger_command.append(fing_msg)
            grip_msg.grasp_mode.mode = gripper_msgs.msg.TesolloGraspingMode.BASIC_MODE
            grip_msg.individual_control = 1
            grip_msg_list.append(grip_msg)
        gripper_cmd_msg.command = grip_msg_list
        self.gripper_cmd_pub_.publish(gripper_cmd_msg)

    
    def timer_callback(self):
        if not self.gripper_callback_status:
            self.get_logger().info(f"Did not receive joint states of gripper... trying again!")
            return
        else:
            rangle = np.deg2rad([-1.9,1.4,63,24.6,
                            -1.9,8.2,64.4,24.3,
                            0.1,-4.1,63.5,25.9])
            #rangle = np.deg2rad(np.zeros((12,)))
            force = [1]*12
            velocity = [np.pi]*12
            self.right_gripper_go(rangle, force=force, speed=velocity)
            time.sleep(3)
            if self.gripper_callback_status:
                #self.gripper_callback_status = False
                self.get_logger().info(f"---------Execution completed.-------")
            
def main(args=None):
    rclpy.init(args=args)
    node = GripperControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
