import ipdb
import rospy
import jax.random as jr
import jax.numpy as jnp

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

from jackal_control.policy import Policy


class JackalMover:

    def __init__(self, path: str, key: jr.PRNGKey):
        rospy.init_node('jackal_controller', anonymous=True)

        # Publisher to control the Jackal
        self.vel_pub = rospy.Publisher('/sparkal1/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)

        # Subscriber to get Jackal's odometry
        self.odom_sub = rospy.Subscriber('/sparkal1/odom', Odometry, self.odom_callback)

        # Initialize robot state
        self.position = [0.0, 0.0]  # (x, y)
        self.orientation = 0.0  # Yaw (in radians)
        self.velocity = [0.0, 0.0]  # (linear x, angular z)

        # Control rate
        self.dt = 0.03
        self.rate = rospy.Rate(int(1 / self.dt))  # Control loop at 10 Hz

        # Create control policy
        self.policy = Policy(path, key)

    def odom_callback(self, msg):
        # Extract position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        # Extract orientation (quaternion)
        orientation_q = msg.pose.pose.orientation
        quaternion = (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        roll, pitch, yaw = euler_from_quaternion(quaternion)

        self.position = [x, y]
        self.orientation = yaw

        self.velocity = [
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z,
        ]

    def action2cmd_vel(self, omega, v) -> Twist:
        cmd_vel = Twist()
        cmd_vel.linear.x = omega
        cmd_vel.angular.z = v
        return cmd_vel

    def run(self):
        while not rospy.is_shutdown():
            # Get graph
            jackal_state = jnp.array([0., 0., 0., 0., 0.])
            jackal_state = jackal_state.at[0].set(self.position[0])
            jackal_state = jackal_state.at[1].set(self.position[1])
            jackal_state = jackal_state.at[2].set(jnp.cos(self.orientation))
            jackal_state = jackal_state.at[3].set(jnp.sin(self.orientation))
            jackal_state = jackal_state.at[4].set(self.velocity[0])
            graph = self.policy.env.get_graph(jackal_state)

            # Compute velocity command using get_control()
            action = self.policy.get_action(graph)
            cmd_vel = self.action2cmd_vel(action[0], action[1])

            # Publish velocity command
            self.vel_pub.publish(cmd_vel)

            # Maintain loop rate
            self.rate.sleep()


if __name__ == '__main__':
    path = "./model/seed0_224120748_ZGJD"

    with ipdb.launch_ipdb_on_exception():
        try:
            controller = JackalMover(path, jr.PRNGKey(0))
            controller.run()
        except rospy.ROSInterruptException:
            pass
