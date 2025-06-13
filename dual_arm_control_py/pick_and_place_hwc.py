import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import gripper_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from dual_arm_control_py.online_joints_trajectory_generator import TrajectoryGenerator
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
        self.gripper_cmd_pub_ = self.create_publisher(gripper_msgs.msg.TesolloCommand,"/svaya/ui/gripper/command",10)
        self.gripper_status_sub_ = self.create_subscription(gripper_msgs.msg.TesolloStatus, "/svaya/ui/gripper/status",self.GripperStatusCallback,10)
        self.gripper_callback_status = False
        self.obj_reach_t = 20
        self.obj_put_t = 10
        self.goal_t = 10
        self.goal_offset_t = 5
        self.return_t = 20
        self.grasp_t = 2
        self.traj_t = self.obj_reach_t +self.obj_put_t + self.goal_offset_t+ self.return_t+ self.grasp_t
        self.grasp_fc = False
        self.grasp_fo = False
        l1,l2, l3, l4, l5 = [0.10555,0.176,0.3,0.32,0.2251]
        self.T_init = np.array([[0, -1, 0, 0], 
                                 [0, 0, -1, -l1-l2-l3-l4-l5], 
                                 [1, 0, 0, 0], 
                                 [0, 0, 0, 1]])
        self.T_obj = np.array([[0, 0.866, 0.5, l4+l5], 
                                 [0, -0.5, 0.866, -0.3], 
                                 [1, 0, 0, -l3], 
                                 [0, 0, 0, 1]])
        self.T_goal = np.array([[0, 0, 1, l4+l5+0.1], 
                                 [0, -1, 0, -0.6], 
                                 [1, 0, 0, -l3], 
                                 [0, 0, 0, 1]])
        self.T_goal_offset = np.array([[0, 0, 1, l4+l5], 
                                 [0, -1, 0, -0.6], 
                                 [1, 0, 0, -l3], 
                                 [0, 0, 0, 1]])

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
        if not self.joint_callback_status:
            self.get_logger().info(f"Did not receive joint states... trying again!")
            return
        else:
            if self.t_counter<self.traj_t:
                if self.t_counter < self.obj_reach_t:
                    angle = self.joint_state_actual        
                    target_angle = self.obj.get_joints(t=self.t_counter, traj_time=self.obj_reach_t, theta=angle[7:14], dh_l=self.obj.dh_l, T_init=self.T_init, T_final=self.T_obj)
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
                elif not self.grasp_fc:
                    if not self.gripper_callback_status:
                        self.get_logger().info(f"Did not receive joint states of gripper... trying again!")
                        return
                    else:
                        rangle = np.deg2rad([-2.9,3.1,86,35.5,
                                    -2.8,11.6,87.7,37.7,
                                    -1.5,-6.4,90,40.2])
                        #rangle = np.deg2rad(np.zeros((12,)))
                        force = [1]*12
                        velocity = [np.pi]*12
                        self.right_gripper_go(rangle, force=force, speed=velocity)
                        time.sleep(self.grasp_t)
                        
                        if not self.grasp_fc:
                            self.grasp_fc = True
                            self.get_logger().info(f"---------Gripper closed.-------")
                elif self.t_counter < self.obj_reach_t+self.obj_put_t:
                    angle = self.joint_state_actual        
                    target_angle = self.obj.get_joints(t=self.t_counter-self.obj_reach_t, traj_time=self.obj_put_t, theta=angle[7:14], dh_l=self.obj.dh_l, T_init=self.T_obj, T_final=self.T_goal)
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
                elif not self.grasp_fo:
                    if not self.gripper_callback_status:
                        self.get_logger().info(f"Did not receive joint states of gripper... trying again!")
                        return
                    else:
                        rangle = np.deg2rad([0.0,0.0,0.0,0.0,
                                            0.0,0.0,0.0,0.0,
                                            0.0,0.0,0.0,0.0])
                        #rangle = np.deg2rad(np.zeros((12,)))
                        force = [1]*12
                        velocity = [np.pi]*12
                        self.right_gripper_go(rangle, force=force, speed=velocity)
                        time.sleep(self.grasp_t)
                        
                        if not self.grasp_fo:
                            self.grasp_fo = True
                            self.get_logger().info(f"---------Gripper Opened.-------")
                elif self.t_counter < self.obj_reach_t+self.obj_put_t+self.goal_offset_t:
                    angle = self.joint_state_actual        
                    target_angle = self.obj.get_joints(t=self.t_counter-(self.obj_reach_t+ self.obj_put_t), traj_time=self.goal_offset_t, theta=angle[7:14], dh_l=self.obj.dh_l, T_init=self.T_goal, T_final=self.T_goal_offset)
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
                elif self.t_counter < self.obj_reach_t+self.obj_put_t+self.goal_offset_t+self.return_t:
                    angle = self.joint_state_actual        
                    target_angle = self.obj.get_joints(t=self.t_counter-(self.obj_reach_t+ self.obj_put_t+self.goal_offset_t), traj_time=self.return_t, theta=angle[7:14], dh_l=self.obj.dh_l, T_init=self.T_goal_offset, T_final=self.T_init)
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
            
                """elif self.t_counter >= (self.obj_reach_t+self.grasp_t) and self.t_counter < (self.obj_reach_t+self.grasp_t+self.obj_put_t):
                    angle = self.joint_state_actual        
                    target_angle = self.obj.get_joints(self.t_counter, angle[7:14], self.obj.dh_l, self.T_obj, self.T_put)
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
                elif self.t_counter >= (self.obj_reach_t+self.grasp_t +self.obj_put_t) and self.t_counter < (self.obj_reach_t+self.grasp_t+self.obj_put_t+self.return_t):
                    angle = self.joint_state_actual        
                    target_angle = self.obj.get_joints(self.t_counter, angle[7:14], self.obj.dh_l, self.T_put, self.T_init)
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
                    self.pub.publish(uicmd_msg)"""

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
                self.pub.publish(uicmd_msg)
                if self.joint_callback_status:
                    self.get_logger().info(f"---------Execution completed.-------")
                    self.joint_callback_status  = False
                
                
    def callback(self, msg):
        self.status = msg
        self.joint_state_actual = np.concatenate((self.status.left_arm.position, self.status.right_arm.position))
        self.joint_callback_status  = True

def main(args=None):
    rclpy.init(args=args)
    node = PositionCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
