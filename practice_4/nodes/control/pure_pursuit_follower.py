#!/usr/bin/env python3
import numpy as np
import rospy
from threading import Lock

from autoware_msgs.msg import Lane, VehicleCmd
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion

from shapely.geometry import LineString, Point
from shapely import prepare, distance
from scipy.interpolate import interp1d

class PurePursuitFollower:
    def __init__(self):
        # Parameters
        self.lock = Lock()
        self.path_linestring = None
        self.distance_to_velocity_interpolator = None
        self.lookahead_distance = rospy.get_param("~lookahead_distance")
        self.wheel_base = rospy.get_param("/wheel_base")
        self.distance_to_goal_limit = rospy.get_param("/lanelet2_global_planner/distance_to_goal_limit")
        # Publishers
        self.vehicle_cmd_pub = rospy.Publisher('/control/vehicle_cmd', VehicleCmd, queue_size=1, tcp_nodelay=True)
        # Subscribers
        rospy.Subscriber('path', Lane, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def path_callback(self, msg):
        if len(msg.waypoints) < 2:
            # If waypoints are empty or one point(edge case) return none values to force stopping commands
            interpolator = None
            path_linestring = None
        else:
            # Create a distance to velocity interpolator for the path with stopping after reaching final point

            # collect waypoint x and y coordinates
            waypoints_xy = np.array([(w.pose.pose.position.x, w.pose.pose.position.y) for w in msg.waypoints])
            # Calculate distances between points
            distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xy, axis=0) ** 2, axis=1)))
            # add 0 distance in the beginning and end
            distances = np.concatenate(([0], distances))
            # Extract velocity values at waypoints
            velocities = np.array([w.twist.twist.linear.x for w in msg.waypoints])
            interpolator = interp1d(distances, velocities, kind='linear', bounds_error=False,
                                    fill_value=(velocities[0], 0))

            # Transform path to line string
            path_linestring = LineString(waypoints_xy)
            prepare(path_linestring)

        with self.lock:
            self.distance_to_velocity_interpolator = interpolator
            self.path_linestring = path_linestring

    def current_pose_callback(self, msg):
        if self.path_linestring is None or self.distance_to_velocity_interpolator is None:
            # Stopping commands
            steering_angle = 0
            linear_velocity = 0
            linear_acceleration = -3
        else:
            # Find lookahead distance
            current_pose = Point([msg.pose.position.x, msg.pose.position.y])
            d_ego_from_path_start = self.path_linestring.project(current_pose)
            lookahead_point = self.path_linestring.interpolate(d_ego_from_path_start + self.lookahead_distance)
            real_lookahead_distance = distance(current_pose, lookahead_point)
            # Find heading and lookahead heading
            _, _, heading = euler_from_quaternion(
                [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
            lookahead_heading = np.arctan2(lookahead_point.y - current_pose.y, lookahead_point.x - current_pose.x)
            # Calculate steering angle and other parameters
            steering_angle = np.arctan(2 * self.wheel_base * np.sin(lookahead_heading - heading) / real_lookahead_distance)
            linear_velocity = self.distance_to_velocity_interpolator(d_ego_from_path_start)
            linear_acceleration = 0

        # Publish command
        vehicle_cmd = VehicleCmd()
        vehicle_cmd.ctrl_cmd.steering_angle = steering_angle
        vehicle_cmd.ctrl_cmd.linear_velocity = linear_velocity
        vehicle_cmd.ctrl_cmd.linear_acceleration = linear_acceleration
        vehicle_cmd.header.frame_id = 'base_link'
        vehicle_cmd.header.stamp = msg.header.stamp

        self.vehicle_cmd_pub.publish(vehicle_cmd)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()
