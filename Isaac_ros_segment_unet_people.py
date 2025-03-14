import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class ObjectAvoidanceNode(Node):
    def __init__(self):
        super().__init__('object_avoidance_node')

        # Subscribe to the segmentation mask topic
        self.segmentation_subscriber = self.create_subscription(
            Image,
            '/unet/colored_segmentation_mask',
            self.segmentation_callback,
            10
        )

        # Publisher for velocity commands
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        self.get_logger().info('Object avoidance node started.')

    def segmentation_callback(self, msg):
        # Convert ROS Image to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Define the range for red color in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        # Convert BGR image to HSV
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Create binary mask where red color is present
        mask = cv2.inRange(hsv_image, lower_red, upper_red)

        # Optionally, check for the red range extending to a second range for better detection
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask = mask | mask2

        # Check if any red objects are detected
        if cv2.countNonZero(mask) > 0:
            self.get_logger().info('Red object (person) detected, moving...')
            self.move_forward()
        else:
            self.get_logger().info('No red objects detected, stopping...')
            self.stop_robot()

    def stop_robot(self):
        # Publish zero velocity to stop the robot
        stop_cmd = Twist()
        self.velocity_publisher.publish(stop_cmd)

    def move_forward(self):
        # Publish a forward movement command
        move_cmd = Twist()
        move_cmd.linear.x = 0.5  # Forward velocity
        move_cmd.angular.z = 0.0  # No rotation
        self.velocity_publisher.publish(move_cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectAvoidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
