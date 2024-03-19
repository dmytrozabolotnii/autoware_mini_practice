#!/usr/bin/env python3

import rospy
import numpy as np

from shapely import MultiPoint
from tf2_ros import TransformListener, Buffer, TransformException
from numpy.lib.recfunctions import structured_to_unstructured
from ros_numpy import numpify, msgify

from sensor_msgs.msg import PointCloud2
from autoware_msgs.msg import DetectedObjectArray, DetectedObject
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point32


BLUE80P = ColorRGBA(0.0, 0.0, 1.0, 0.8)

class ClusterDetector:
    def __init__(self):
        self.min_cluster_size = rospy.get_param('~min_cluster_size')
        self.output_frame = rospy.get_param('/detection/output_frame')
        self.transform_timeout = rospy.get_param('~transform_timeout')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.objects_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('points_clustered', PointCloud2, self.cluster_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())

    def cluster_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z', 'label']], dtype=np.float32)

        result_object_array = DetectedObjectArray()
        result_object_array.header.stamp = msg.header.stamp
        result_object_array.header.frame_id = self.output_frame

        if len(points) == 0:
            self.objects_pub.publish(result_object_array)
            return

        if msg.header.frame_id != self.output_frame:
            # fetch transform for target frame
            try:
                transform = self.tf_buffer.lookup_transform(self.output_frame, msg.header.frame_id, msg.header.stamp,
                                                            rospy.Duration(self.transform_timeout))
            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                return

            tf_matrix = numpify(transform.transform).astype(np.float32)
            # make copy of points
            points_copy = points.copy()
            # turn into homogeneous coordinates
            points_copy[:, 3] = 1
            # transform points to target frame
            points_copy = points_copy.dot(tf_matrix.T)
            # write converted coordinates back
            points[:, :3] = points_copy[:, :3]

        for i in range(int(max(points[:, 3]) + 1)):
            obj = DetectedObject()
            obj.header.stamp = msg.header.stamp
            obj.header.frame_id = self.output_frame
            # Find all points with a label
            points3d = points[points[:, 3] == i, :3]
            # Filter out min cluster size
            if len(points3d) < self.min_cluster_size:
                continue
            # Calculate centroids
            centroid_x = np.mean(points3d[:, 0])
            centroid_y = np.mean(points3d[:, 1])
            centroid_z = np.mean(points3d[:, 2])

            # create convex hull
            points_2d = MultiPoint(points3d[:, :2])
            hull = points_2d.convex_hull
            convex_hull_points = [Point32(x, y, centroid_z) for x, y in hull.exterior.coords]

            # Write to object
            obj.pose.position.x = centroid_x
            obj.pose.position.y = centroid_y
            obj.pose.position.z = centroid_z
            obj.convex_hull.polygon.points = convex_hull_points

            obj.id = i
            obj.label = "unknown"
            obj.color = BLUE80P
            obj.valid = True
            obj.space_frame = self.output_frame
            obj.pose_reliable = True
            obj.velocity_reliable = False
            obj.acceleration_reliable = False

            result_object_array.objects.append(obj)

        self.objects_pub.publish(result_object_array)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('cluster_detector', log_level=rospy.INFO)
    node = ClusterDetector()
    node.run()
