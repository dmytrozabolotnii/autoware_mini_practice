#!/usr/bin/env python3
import numpy as np
import rospy

from autoware_msgs.msg import Lane, VehicleCmd
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion

from shapely.geometry import LineString, Point
from shapely import prepare, distance

class PurePursuitFollower:
    def __init__(self):

        # Parameters
        self.path_linestring = None
        self.lookahead_distance = rospy.get_param("~lookahead_distance")
        self.wheel_base = rospy.get_param("/wheel_base")
        # Publishers
        self.vehicle_cmd_pub = rospy.Publisher('/control/vehicle_cmd', VehicleCmd, queue_size=1, tcp_nodelay=True)
        # Subscribers
        rospy.Subscriber('path', Lane, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def path_callback(self, msg):
        path_linestring = LineString([(w.pose.pose.position.x, w.pose.pose.position.y) for w in msg.waypoints])
        prepare(path_linestring)
        self.path_linestring = path_linestring

    def current_pose_callback(self, msg):
        if self.path_linestring is not None:
            # Find lookahead distance
            current_pose = Point([msg.pose.position.x, msg.pose.position.y])
            d_ego_from_path_start = self.path_linestring.project(current_pose)
            lookahead_point = self.path_linestring.interpolate(d_ego_from_path_start + self.lookahead_distance)
            real_lookahead_distance = max(distance(current_pose, lookahead_point), 5)
            # print('lookahead', real_lookahead_distance)
            # Find heading and lookahead heading
            _, _, heading = euler_from_quaternion(
                [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
            lookahead_heading = np.arctan2(lookahead_point.y - current_pose.y, lookahead_point.x - current_pose.x)
            # Calculate steering angle
            steering_angle = np.arctan(2 * self.wheel_base * (lookahead_heading - heading) / real_lookahead_distance)

            # Publish command
            vehicle_cmd = VehicleCmd()
            vehicle_cmd.ctrl_cmd.steering_angle = steering_angle
            vehicle_cmd.ctrl_cmd.linear_velocity = 10.0
            vehicle_cmd.header.frame_id = 'base_link'
            vehicle_cmd.header.stamp = msg.header.stamp

            self.vehicle_cmd_pub.publish(vehicle_cmd)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()