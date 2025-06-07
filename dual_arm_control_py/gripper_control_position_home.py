import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class GripperPositionPublisher(Node):
    def __init__(self):
        super().__init__('gripper_position_publisher')
        self.publisher_ = self.create_publisher(JointTrajectory, '/gripper_position_controller/joint_trajectory', 10)
        #self.send_goal()
        self.timer = self.create_timer(1, self.send_goal)

    def send_goal(self):
        traj = JointTrajectory()
        traj.joint_names = ["L_F1M1", "L_F1M2", "L_F1M3", "L_F1M4", "L_F2M1", "L_F2M2", "L_F2M3", "L_F2M4", "L_F3M1", "L_F3M2", "L_F3M3", "L_F3M4", 
                            "R_F1M1", "R_F1M2", "R_F1M3", "R_F1M4", "R_F2M1", "R_F2M2", "R_F2M3", "R_F2M4", "R_F3M1", "R_F3M2", "R_F3M3", "R_F3M4"]  # Replace with your actual joint name
        point = JointTrajectoryPoint()
        pot = [0.0 for _ in range(24)]
        pot[12] = 0.9
        point.positions = pot
        point.time_from_start.sec = 1
        traj.points.append(point)
        self.publisher_.publish(traj)
        self.get_logger().info('Sent gripper position command')

def main(args=None):
    rclpy.init(args=args)
    node = GripperPositionPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
