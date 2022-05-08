import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import math
from tf.transformations import euler_from_quaternion

class control_turtlebot():
    def __init__(self):
        rospy.init_node('TurtleControl', anonymous=True)
        rospy.on_shutdown(self.shutdown)
        
        self.temp_odom = None
        self.cmd_vel_object = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.vel_msg = Twist()
        self.vel_msg.linear.x = 1
        self.vel_msg.angular.z = math.pi/8

        self.rate = 3
        self.duration = 30

        r = rospy.Rate(self.rate)

        self.logged_data = []

        while not rospy.is_shutdown():
            self.cmd_vel_object.publish(self.vel_msg)

            if self.temp_odom:
                self.logged_data.append(self.temp_odom)

                if len(self.logged_data) >= self.rate*self.duration:
                    rospy.signal_shutdown('Finished collecting data')
                    self.plot_poses()

            r.sleep()

    def odom_callback(self, msg):
        self.temp_odom = msg
        # print(msg.header.stamp)
        # print(msg.pose.pose.position)

    def plot_poses(self):
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        x, y, u, v = [], [], [], []
        scale = 0.1
        for msg in self.logged_data:
            x.append(msg.pose.pose.position.x)
            y.append(msg.pose.pose.position.y)

            o = msg.pose.pose.orientation
            q = [o.x, o.y, o.z, o.w]
            (roll, pitch, yaw) = euler_from_quaternion(q)
            u.append(math.cos(yaw)*scale)
            v.append(math.sin(yaw)*scale)

        plt.quiver(x, y, u, v, width=0.002)
        plt.show()

    def shutdown(self):
        self.cmd_vel_object.publish(Twist())
        rospy.loginfo('Node shutdown!')

        rospy.sleep(1)

if __name__ == '__main__':
    try:
        control_turtlebot()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(type(e))
        rospy.loginfo('Node Terminated')

        raise e

