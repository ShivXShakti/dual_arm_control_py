from darm_api.darm_api import DarmAPI
import darm_msgs.msg
import numpy as np
import rclpy 
import time 
by_time = darm_msgs.msg.PlanningMethod.BY_TIME
by_vmax = darm_msgs.msg.PlanningMethod.BY_MAX_ACC_VEL 

def main(args=None):
    rclpy.init(args=args)
    try:
        api = DarmAPI()
        langle = np.deg2rad([-58,0,8,0,
                            -1.6,0,30,-2,
                            -1.1,0,30,-2])
        rangle = np.deg2rad([58,0,8,0,
                            -1.6,0,30,-2,
                            -1.1,0,30,-2])
        force = [1]*12
        velocity = [np.pi]*12
        print('obj craeated ')
        api.left_gripper_go(langle, force=force,speed = velocity)
        api.right_gripper_go(rangle,force=force,speed = velocity)
        
    except Exception as e:
        print(f"Error in the node: {e}")

    rclpy.shutdown()

if __name__ == '__main__':
    main()














