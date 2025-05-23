import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
from python_pkg.traj_generator_ik import TrajectoryGenerator
from sensor_msgs.msg import JointState

class PositionCommander(Node):
    def __init__(self):
        super().__init__('position_commander')
        self.sampling_frequency = 100
        self.publisher_ = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        self.timer = self.create_timer(1/self.sampling_frequency, self.timer_callback)  # 100 Hz
        self.start_time = self.get_clock().now().nanoseconds
        self.obj = TrajectoryGenerator()
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10
        )

        # Number of joints expected
        self.num_joints = 16  # Adjust according to your robot
        self.js = None
        self.joint_names = [
            'J1_left', 'J2_left', 'J3_left', 'J4_left', 'J5_left', 'J6_left', 'J7_left',
            'J1_right', 'J2_right', 'J3_right', 'J4_right', 'J5_right', 'J6_right', 'J7_right',
            'neck_joint', 'head_joint'
        ]
        self.ordered_positions = None #[0.0 for _ in range(len(self.joint_names))]
        self.t_counter = 0.0

    def listener_callback(self, msg):
        self.get_logger().info('--- Received Joint States ---')
        #self.js = msg.position
        name_to_position = dict(zip(msg.name, msg.position))
        self.ordered_positions = [name_to_position.get(self.joint_names[i], float('nan')) for i in range(len(self.joint_names))]
        self.get_logger().info(f"ordered joints: {self.ordered_positions}")
    
    def timer_callback(self):
        #t = (self.get_clock().now().nanoseconds - self.start_time) * 1e-9
        if self.ordered_positions is None:
            return
        msg = Float64MultiArray()
        joints = self.obj.get_joints(self.t_counter, [self.ordered_positions[7], self.ordered_positions[8], self.ordered_positions[9], self.ordered_positions[10], self.ordered_positions[11], self.ordered_positions[12], self.ordered_positions[13]], self.obj.dh_l)
        # Fill with sinusoidal dummy values
        msg.data = [joints[i] for i in range(16)]
        #msg.data = [0.2 for i in range(16)]
        #msg.data = self.obj.get_joints(self.t_counter, [self.ordered_positions[7], self.ordered_positions[8], self.ordered_positions[9], self.ordered_positions[10], self.ordered_positions[11], self.ordered_positions[12], self.ordered_positions[13]])
        self.get_logger().info(f"timer callback: {msg.data}")
        #msg.data = [0.5 * math.sin(t + i) for i in range(self.num_joints)]
        self.t_counter += 1/self.sampling_frequency
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PositionCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
