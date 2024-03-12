#!/usr/bin/env python3
import numpy as np
import rospy

from geometry_msgs.msg import PoseStamped
from autoware_msgs.msg import Lane, Waypoint
# All these imports from lanelet2 library should be sufficient
import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest


class GlobalPlanner:
    def __init__(self):
        # Parameters
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")
        self.speed_limit = float(rospy.get_param("~speed_limit"))

        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        self.output_frame = rospy.get_param("/lanelet2_global_planner/output_frame")
        self.distance_to_goal_limit = rospy.get_param("/lanelet2_global_planner/distance_to_goal_limit")
        # Load the map using Lanelet2
        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise RuntimeError('Only "utm" is supported for lanelet2 map loading')
        self.lanelet2_map = load(lanelet2_map_name, projector)
        self.current_location = None
        self.goal_point = None

        # traffic rules
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                      lanelet2.traffic_rules.Participants.VehicleTaxi)
        # routing graph
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)

        # Publishers
        self.global_path_pub = rospy.Publisher('/global_path', Lane, latch=True, queue_size=1, tcp_nodelay=True)
        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def goal_callback(self, msg):
        # Current location check
        if self.current_location is None:
            return

        self.goal_point = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                      msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                      msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                      msg.pose.orientation.w, msg.header.frame_id)

        # get start and end lanelets
        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]

        # find routing graph
        route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)
        if route is None:
            rospy.logwarn("%s - No route is found to goal position", rospy.get_name())
            return

        # find shortest path
        path = route.shortestPath()
        if path is None:
            rospy.logwarn("%s - No path is found to goal position", rospy.get_name())
            return

        # this returns LaneletSequence to a point where lane change would be necessary to continue
        path_no_lane_change = path.getRemainingLane(start_lanelet)

        waypoints_list = self.convert_laneletseq_to_waypoints_list(path_no_lane_change)

        self.publish_lane_from_waypoints_list(waypoints_list)

    def current_pose_callback(self, msg):
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        if self.goal_point is None:
            return

        # Check if we need to stop and push empty global planner
        if ((self.current_location.x - self.goal_point.x) ** 2 +
           (self.current_location.y - self.goal_point.y) ** 2) ** 0.5 < self.distance_to_goal_limit:
            self.publish_lane_from_waypoints_list([])
            self.goal_point = None
            rospy.loginfo("%s - goal position reached, cleared path", rospy.get_name())

    def convert_laneletseq_to_waypoints_list(self, laneletseq):
        waypoints = []
        min_distance_to_goal_endpoint = np.inf
        closest_waypoint = None
        for j, lanelet in enumerate(laneletseq):
            # Obtain speed from lanelet or global speed
            if 'speed_ref' in lanelet.attributes:
                speed = min(float(lanelet.attributes['speed_ref']) / 3.6, self.speed_limit / 3.6)
            else:
                speed = self.speed_limit
            for i, point in enumerate(lanelet.centerline):
                # Check and omit for first waypoint of every lanelet except very first lanelet
                if i == 0 and j != 0:
                    continue
                waypoint = Waypoint()
                waypoint.pose.pose.position.x = point.x
                waypoint.pose.pose.position.y = point.y
                waypoint.pose.pose.position.z = point.z
                waypoint.twist.twist.linear.x = speed
                waypoints.append(waypoint)

                # Find the waypoint closest to endpoint in the last lanelet
                if j == len(laneletseq) - 1:
                    distance_to_goal_endpoint = ((point.x - self.goal_point.x) ** 2 + (point.y - self.goal_point.y) ** 2) ** 0.5
                    if distance_to_goal_endpoint < min_distance_to_goal_endpoint:
                        min_distance_to_goal_endpoint = distance_to_goal_endpoint
                        closest_waypoint = waypoint

        # Update goal point and return shortened list of waypoints
        if closest_waypoint is not None:
            self.goal_point = BasicPoint2d(closest_waypoint.pose.pose.position.x, closest_waypoint.pose.pose.position.y)

            return waypoints[:waypoints.index(closest_waypoint) + 1]
        else:
            return waypoints

    def publish_lane_from_waypoints_list(self, waypoints):
        # Publishing function for waypoints list to global planner lane
        lane = Lane()
        lane.header.frame_id = self.output_frame
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = waypoints

        self.global_path_pub.publish(lane)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('global_planner')
    node = GlobalPlanner()
    node.run()
