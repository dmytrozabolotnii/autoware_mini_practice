#!/usr/bin/env python3

import math

import rospy
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from sklearn.cluster import DBSCAN

from ros_numpy import numpify, msgify
from sensor_msgs.msg import PointCloud2


class PointsClusterer:
    def __init__(self):
        # Parameters
        self.cluster_epsilon = rospy.get_param('points_clusterer/cluster_epsilon')
        self.cluster_min_size = rospy.get_param('points_clusterer/cluster_min_size')
        self.clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=self.cluster_min_size)
        # Publishers
        self.clustered_pub = rospy.Publisher('points_clustered', PointCloud2, queue_size=1, tcp_nodelay=True)
        # Subscribers
        rospy.Subscriber('points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2 ** 24,
                         tcp_nodelay=True)
        rospy.loginfo("%s - initialized", rospy.get_name())

    def points_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        # Clustering
        labelled = self.clusterer.fit_predict(points)
        assert len(labelled) == len(points), (rospy.get_name() +
                                              'raw points and clustered points should be the same size')
        # Stacking
        points_labeled = np.concatenate((points, labelled[:, np.newaxis]), axis=1)
        # Filtering out non-clustered noise
        points_labeled = points_labeled[points_labeled[:, 3] != -1]
        # convert labelled points to PointCloud2 format
        data = unstructured_to_structured(points_labeled, dtype=np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('label', np.int32)
        ]))
        # publish clustered points message
        cluster_msg = msgify(PointCloud2, data)
        cluster_msg.header.stamp = msg.header.stamp
        cluster_msg.header.frame_id = msg.header.frame_id

        self.clustered_pub.publish(cluster_msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('points_clusterer', log_level=rospy.INFO)
    node = PointsClusterer()
    node.run()
