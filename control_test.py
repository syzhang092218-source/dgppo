import rospy
from geometry_msgs.msg import Twist
import time


def move_jackal():
    rospy.init_node('jackal_mover', anonymous=True)
    vel_pub = rospy.Publisher('/sparkal1/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    vel_msg = Twist()
    vel_msg.linear.x = 0.3  # Move forward
    vel_msg.angular.z = 0.0  # No rotation

    start_time = time.time()
    duration = 4

    while time.time() - start_time < duration and not rospy.is_shutdown():
        vel_pub.publish(vel_msg)
        rate.sleep()

    # Stop the robot after the duration
    vel_msg.linear.x = 0.0
    vel_pub.publish(vel_msg)


if __name__ == '__main__':
    try:
        move_jackal()
    except rospy.ROSInterruptException:
        pass
