#!/usr/bin/env python3
import darm_msgs.msg
import gripper_msgs.msg
import fts_msgs.msg
import rclpy
from rclpy.node import Node
import numpy as np
import time 
import threading


class DarmAPI(Node):
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
        

        self.ui_cmd_pub_ = self.create_publisher(darm_msgs.msg.UiCommand,"/svaya/ui/command",10)
        self.ui_status_sub_ = self.create_subscription(darm_msgs.msg.UiStatus,"svaya/ui/status",self.UiStatusCallback,10)
        self.gripper_cmd_pub_ = self.create_publisher(gripper_msgs.msg.TesolloCommand,"/svaya/ui/gripper/command",10)
        self.gripper_status_sub_ = self.create_subscription(gripper_msgs.msg.TesolloStatus, "/svaya/ui/gripper/status",self.GripperStatusCallback,10)
        self.ui_fts_pub_ = self.create_publisher(fts_msgs.msg.FtsBias,"svaya/fts/bias",10)
        self.ui_fts_sub_ = self.create_subscription(fts_msgs.msg.FtsData,"svaya/fts/status",self.FTSCallback,10)

        self.thread = threading.Thread(target=self.ros_spin)
        self.thread.start()
        time.sleep(3)
    def ros_spin(self):
        while(rclpy.ok()):
            rclpy.spin_once(self)
            time.sleep(0.001)
        self.destroy_node()
    def robot_cmd_publish(self,msg):
        if not self.stop_robot_:
            self.ui_cmd_pub_.publish(msg)

    def GripperStatusCallback(self,msg):
        self.gripper_status_ = msg

    def FTSCallback(self,msg: fts_msgs.msg.FtsData):
        self.fts_status_= msg

    def UiStatusCallback(self,msg):
        self.left_joint_status_ = msg.left_arm.position
        self.right_joint_status_ = msg.right_arm.position
        self.head_joint_status_ = msg.head.position

        self.la_in_exe_ = msg.left_arm_in_execution
        self.ra_in_exe_ = msg.right_arm_in_execution
        self.h_in_exe_ = msg.head_in_execution
        if (msg.stop_robot):
            self.stop_robot_ = msg.stop_robot

    def left_arm_go(self,joint_angle,by_time_vmax=0,time_speed=3):
        cmd_msg = darm_msgs.msg.UiCommand()
        if len(joint_angle) != 7:
            print("wrong joint angle")
            return
        cmd_msg.left_arm = list(joint_angle)
        cmd_msg.part.id = darm_msgs.msg.RobotPart.LEFT_ARM 
        cmd_msg.time_to_reach = 3.0
        cmd_msg.vmax = np.pi/4
        if by_time_vmax == 0:
            cmd_msg.planning_method.id = darm_msgs.msg.PlanningMethod.BY_TIME
            if (time_speed<0.5):
                print("time is not correct")
                return
            cmd_msg.time_to_reach = float(time_speed)
        elif by_time_vmax == 1:
            cmd_msg.planning_method.id = darm_msgs.msg.PlanningMethod.BY_MAX_ACC_VEL
            cmd_msg.vmax = float(abs(time_speed))
        else:
            print("wrong planning method")
            return
        self.robot_cmd_publish(cmd_msg)
        

    def right_arm_go(self,joint_angle,by_time_vmax=0,time_speed=3):
        cmd_msg = darm_msgs.msg.UiCommand()
        if len(joint_angle) != 7:
            print("wrong joint angle")
            return
        cmd_msg.right_arm = list(joint_angle)
        cmd_msg.part.id = darm_msgs.msg.RobotPart.RIGHT_ARM 
        cmd_msg.time_to_reach = 3.0
        cmd_msg.vmax = np.pi/4
        if by_time_vmax ==0:
            if (time_speed<0.5):
                print("time is not correct")
                return
            cmd_msg.planning_method.id = darm_msgs.msg.PlanningMethod.BY_TIME
            cmd_msg.time_to_reach = float(time_speed)
        elif by_time_vmax ==1:
            cmd_msg.planning_method.id = darm_msgs.msg.PlanningMethod.BY_MAX_ACC_VEL
            cmd_msg.vmax = float(abs(time_speed))
        else:
            print("wrong planning method")
            return
        self.robot_cmd_publish(cmd_msg)


        
    def head_go(self,joint_angle,by_time_vmax=0,time_speed=3):
        cmd_msg = darm_msgs.msg.UiCommand()
        if len(joint_angle) != 2:
            print("wrong joint angle")
            return
        cmd_msg.head = list(joint_angle)
        cmd_msg.part.id = darm_msgs.msg.RobotPart.HEAD 
        cmd_msg.time_to_reach = 3.0
        cmd_msg.vmax = np.pi/4
        if by_time_vmax ==0:
            cmd_msg.planning_method.id = darm_msgs.msg.PlanningMethod.BY_TIME
            if (time_speed<0.5):
                print("time is not correct")
                return
            cmd_msg.time_to_reach = float(abs(time_speed))
        elif by_time_vmax ==1:
            cmd_msg.planning_method.id = darm_msgs.msg.PlanningMethod.BY_MAX_ACC_VEL
            cmd_msg.vmax = time_speed
        else:
            print("wrong planning method")
            return
        self.robot_cmd_publish(cmd_msg)

        
    def both_arm_go(self,joint_angle,by_time_vmax=0,time_speed=3):
        cmd_msg = darm_msgs.msg.UiCommand()
        if len(joint_angle) != 14:
            print("wrong joint angle")
            return
        cmd_msg.left_arm = list(joint_angle[:7])
        cmd_msg.right_arm = list(joint_angle[7:14])
        cmd_msg.part.id = darm_msgs.msg.RobotPart.BOTH_ARM 
        cmd_msg.time_to_reach = 3.0
        cmd_msg.vmax = np.pi/4
        if by_time_vmax ==0:
            cmd_msg.planning_method.id = darm_msgs.msg.PlanningMethod.BY_TIME
            cmd_msg.time_to_reach = float(abs(time_speed))
            if (time_speed<0.5):
                print("time is not correct")
                return
        elif by_time_vmax ==1:
            cmd_msg.planning_method.id = darm_msgs.msg.PlanningMethod.BY_MAX_ACC_VEL
            cmd_msg.vmax = float(abs(time_speed))
        else:
            print("wrong planning method")
            return
        self.robot_cmd_publish(cmd_msg)

        
    def robot_go(self,joint_angle,by_time_vmax=0,time_speed=3):
        cmd_msg = darm_msgs.msg.UiCommand()
        if len(joint_angle) != 16:
            print("wrong joint angle")
            return
        cmd_msg.left_arm = list(joint_angle[:7])
        cmd_msg.right_arm = list(joint_angle[7:14])
        cmd_msg.head = list(joint_angle[14:16])
        cmd_msg.part.id = darm_msgs.msg.RobotPart.WHOLE_BODY 
        cmd_msg.time_to_reach = 3.0
        cmd_msg.vmax = np.pi/4
        if by_time_vmax ==0:
            cmd_msg.planning_method.id = darm_msgs.msg.PlanningMethod.BY_TIME
            cmd_msg.time_to_reach = float(abs(time_speed))
        elif by_time_vmax ==1:
            cmd_msg.planning_method.id = darm_msgs.msg.PlanningMethod.BY_MAX_ACC_VEL
            cmd_msg.vmax = float(abs(time_speed))
        else:
            print("wrong planning method")
            return
        self.robot_cmd_publish(cmd_msg)

    
    def wait_for_left_arm(self):
        time.sleep(0.1)
        while(self.la_in_exe_):
            time.sleep(0.001)

    def wait_for_right_arm(self):
        time.sleep(0.1)
        while(self.ra_in_exe_):
            time.sleep(0.001)

    def wait_for_head(self):
        time.sleep(0.1)
        while(self.h_in_exe_):
            time.sleep(0.001)

    def wait_for_both_arm(self):
        time.sleep(0.1)
        while(self.la_in_exe_  or self.ra_in_exe_):
            time.sleep(0.001)

    def wait_for_whole_body(self):
        time.sleep(0.1)
        while(self.la_in_exe_  or self.ra_in_exe_ or self.h_in_exe_):
            time.sleep(0.001)
            
    
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

    
    def both_gripper_go(self,gripper_motor_angle,force = np.zeros((24,)),speed = np.zeros((24,))):
        if len(gripper_motor_angle) != 24:
            print("wrong gripper joint entry: the current size is: ", len(gripper_motor_angle))
            return
        self.left_gripper_go(gripper_motor_angle[:12],speed=speed[:12],force=force[:12])
        self.right_gripper_go(gripper_motor_angle[12:24],speed=speed[12:24],force=force[12:24])


        
    def set_fts_bias(self,bias_command,sampling_command=[3,3],low_pass_filter=[8,8]):
        msg =  fts_msgs.msg.FtsBias()
        msg.fts_bias_command = bias_command
        msg.fts_sampling_commad = sampling_command
        msg.fts_low_pass_filter_command = low_pass_filter
        self.ui_fts_pub_.publish(msg)
        
    def get_left_fts_data(self):
        return self.fts_status_.data[0]
    
    def get_right_fts_data(self):
        return self.fts_status_.data[1]
   
