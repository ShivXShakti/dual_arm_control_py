from darm_api.darm_api import DarmAPI
import darm_msgs.msg
import numpy as np
import rclpy 
import time 
import threading
by_time = darm_msgs.msg.PlanningMethod.BY_TIME
by_vmax = darm_msgs.msg.PlanningMethod.BY_MAX_ACC_VEL 



def left_arm_fun(api):
    angle = np.deg2rad(np.zeros((12,)))
    api.left_gripper_go(angle)
    print("HIHI")
    api.right_gripper_go(angle)
    angle = np.deg2rad([0,-10])
    api.head_go(angle,by_time_vmax=by_time, time_speed=5)

    while(rclpy.ok() and not api.stop_robot_):
        """angle = np.deg2rad(np.zeros((12,)))
        api.left_gripper_go(angle)
        time.sleep(2)
        angle = np.deg2rad(np.zeros((12,)))
        api.right_gripper_go(angle)
        time.sleep(2)"""
        langle = np.deg2rad([-58,0,8,0,
                            -1.6,0,30,-2,
                            -1.1,0,30,-2])
        rangle = np.deg2rad([58,0,8,0,
                            -1.6,0,30,-2,
                            -1.1,0,30,-2])
        force = [1]*12
        velocity = [np.pi]*12
        api.left_gripper_go(langle, force=force,speed = velocity)
        api.right_gripper_go(rangle,force=force,speed = velocity)
        time.sleep(60)
        angle = np.deg2rad([0,0])
        api.head_go(angle,by_time_vmax=by_time, time_speed=5)
        print("HIHIHIHIHII")
     

    
def main(args=None):
    rclpy.init(args=args)
    try:
        api = DarmAPI()
        print('obj craeated ')
        t1  = threading.Thread(target=left_arm_fun,args=(api,))
        # t2 = threading.Thread(target=right_arm_fun,args=(api,))
        t1.start()
        # t2.start()

        t1.join()
        # t2.join()
        
    except Exception as e:
        print(f"Error in the node: {e}")

    rclpy.shutdown()

if __name__ == '__main__':
    main()














