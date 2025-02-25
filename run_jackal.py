import ipdb
import rospy
import jax.random as jr
import jax.numpy as jnp

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from tf.transformations import euler_from_quaternion

from jackal_control.policy import Policy


class JackalMover:

    def __init__(self, path: str, key: jr.PRNGKey):
        rospy.init_node('jackal_controller', anonymous=True)

        # Publisher to control the Jackal
        self.vel_pub = rospy.Publisher('/sparkal2/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)

        # Subscriber to get Jackal's odometry
        self.odom_sub = rospy.Subscriber('/sparkal2/odom', Odometry, self.odom_callback)

        # Publisher of the current orientation
        # self.orientation_pub = rospy.Publisher('/sparkal1/jackal_orientation', float, queue_size=10)

        # Publisher of the current position
        self.position_pub = rospy.Publisher('/sparkal2/jackal_position', Odometry, queue_size=10)

        # Initialize robot state
        self.position = [0.0, 0.0]  # (x, y)
        self.orientation = 0.0  # Yaw (in radians)
        self.velocity = [0.0, 0.0]  # (linear x, angular z)

        # Control rate
        self.dt = 0.02
        self.rate = rospy.Rate(int(1 / self.dt))

        # Create control policy
        policy_key, goal_key, self.key = jr.split(key, 3)
        self.policy = Policy(path, policy_key)

        # Create goal
        self.goals = jnp.array([
            [1.5, 0],
            [1.5, 1.5],
            [0, 1.5],
            [0, 0],
        ])
        self.goal_id = 0
        # self.init_graph = self.policy.env.reset(goal_key)

        # Recode odometry offset
        self.odom_offset = [0.0, 0.0]
        self.orientation_offset = 0.0

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
        self.orientation = jnp.arctan2(jnp.sin(yaw), jnp.cos(yaw))

        self.velocity = [
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z,
        ]

    def action2cmd_vel(self, omega, v) -> Twist:
        cmd_vel = Twist()
        cmd_vel.linear.x = v
        cmd_vel.angular.z = omega
        return cmd_vel

    def run(self):
        # Calibration
        self.odom_offset = self.position
        self.orientation_offset = self.orientation

        while not rospy.is_shutdown():
            # Get goal position
            # goal_pos = self.init_graph.type_states(type_idx=1, n_type=1)[:, :2]
            goal_pos = self.goals[self.goal_id % 4]

            # Get Jackal state
            jackal_state = jnp.array([0., 0., 0., 0., 0.])
            jackal_state = jackal_state.at[0].set(self.position[0] - self.odom_offset[0])
            jackal_state = jackal_state.at[1].set(self.position[1] - self.odom_offset[1])
            jackal_state = jackal_state.at[2].set(jnp.cos(self.orientation - self.orientation_offset))
            jackal_state = jackal_state.at[3].set(jnp.sin(self.orientation - self.orientation_offset))
            jackal_state = jackal_state.at[4].set(self.velocity[0])

            # Publish position
            odom_msg = Odometry()
            odom_msg.pose.pose.position.x = jackal_state[0]
            odom_msg.pose.pose.position.y = jackal_state[1]
            odom_msg.pose.pose.position.z = 0.0
            odom_msg.pose.pose.orientation.x = 0.0
            odom_msg.pose.pose.orientation.y = 0.0
            odom_msg.pose.pose.orientation.z = 0.0
            odom_msg.pose.pose.orientation.w = self.orientation - self.orientation_offset
            self.position_pub.publish(odom_msg)

            # Get graph
            graph = self.policy.create_graph(jackal_state[None], goal_pos)

            # Compute velocity command using get_control()
            action = self.policy.get_action(graph)
            cmd_vel = self.action2cmd_vel(action[0], action[1])

            # Publish velocity command
            self.vel_pub.publish(cmd_vel)
            # self.orientation_pub.publish(self.orientation)

            # See if reach
            dist2goal = jnp.linalg.norm(goal_pos - jackal_state[:2])
            if dist2goal < 0.1:
                print(f'Goal {self.goal_id} reached!')
                # Get a new goal
                # goal_key, self.key = jr.split(self.key)
                # self.init_graph = self.policy.env.reset(goal_key)
                self.odom_offset = [self.position[0] - self.goals[self.goal_id % 4][0],
                                    self.position[1] - self.goals[self.goal_id % 4][1]]
                self.goal_id += 1

            # Maintain loop rate
            self.rate.sleep()


if __name__ == '__main__':
    path = "./model/goal_reaching2"

    with ipdb.launch_ipdb_on_exception():
        try:
            controller = JackalMover(path, jr.PRNGKey(0))
            controller.run()
        except rospy.ROSInterruptException:
            pass
