import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class GripperEffortPublisher(Node):
    def __init__(self):
        super().__init__('gripper_effort_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/gripper_position_controller/commands', 10)
        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.send_effort)

    def send_effort(self):
        msg = Float64MultiArray()
        msg.data = [0.9]*24  # Example effort value
        self.publisher_.publish(msg)
        self.get_logger().info('Published effort command: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    node = GripperEffortPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
