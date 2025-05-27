import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math

class PositionCommander(Node):
    def __init__(self):
        super().__init__('position_commander')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        self.timer = self.create_timer(0.01, self.timer_callback)  # 100 Hz
        self.start_time = self.get_clock().now().nanoseconds

        # Number of joints expected
        self.num_joints = 16  # Adjust according to your robot

    def timer_callback(self):
        t = (self.get_clock().now().nanoseconds - self.start_time) * 1e-9

        msg = Float64MultiArray()
        # Fill with sinusoidal dummy values
        msg.data = [0.0 for i in range(self.num_joints)]

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PositionCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
