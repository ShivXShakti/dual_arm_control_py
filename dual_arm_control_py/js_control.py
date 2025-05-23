import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math
import time

class JointMover(Node):
    def __init__(self):
        super().__init__('joint_mover')
        self.publisher_ = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        #self.start_time = time.time()
        self.start_time_ns = self.get_clock().now().nanoseconds


        # Define joint names (must match names in your URDF)
        self.joint_names = ['neck_joint', 'head_joint',
                            'J1_right', 'J2_right', 'J3_right', 'J4_right', 'J5_right', 'J6_right', 'J7_right',
                            'J1_left', 'J2_left', 'J3_left', 'J4_left', 'J5_left', 'J6_left', 'J7_left'
                             ]
        #self.joint_state = JointState()
        #self.joint_state.name = self.joint_names
        #self.joint_state.position = [0.0 for _ in self.joint_names]

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        #t = time.time() - self.start_time
        t = (self.get_clock().now().nanoseconds - self.start_time_ns) * 1e-9


        msg = JointState()
        msg.header.stamp = now
        msg.name = self.joint_names
        """msg.position = [
            0.0,
            0.5 * math.sin(t),
            0.5 * math.sin(t + 1),
            0.5 * math.sin(t + 2),
            0.5 * math.sin(t),
            0.5 * math.sin(t),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0
        ]"""
        msg.position = [
            0.5,0.5,
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,0.0,0.0
        ]

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointMover()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
