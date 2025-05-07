#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
import open3d as o3d

class Preprocess:
    def __init__(self):
        self.output_topic = "/downsampled_np_pointcloud"
        self.voxel_size = 0.02
        self.pc_pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=10)
        
        # 初始化变换矩阵
        self.translation_vector = np.array([0.0, 0.0, 0.19176])#tx, ty, tz 
        yaw, pitch, roll = 102.54, 179.67, 87.5975
        self.rotation_matrix = self.euler_to_rotation_matrix(roll, pitch, yaw)

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """
        将欧拉角转换为旋转矩阵。
        """
        roll, pitch, yaw = np.radians(roll), np.radians(pitch), np.radians(yaw)
        R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def apply_extrinsic_correction(self, cloud_np):
        """
        应用外参矫正。
        """
        return np.dot(cloud_np, self.rotation_matrix.T) + self.translation_vector

    def publish_pointcloud(self, data, frame_id):
        """
        发布点云数据。
        """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1)
        ]

        pc_msg = point_cloud2.create_cloud(header, fields, data)
        self.pc_pub.publish(pc_msg)
        # rospy.loginfo("降采样后的点云已发布到话题: %s", self.output_topic)

    def downsample_pointcloud(self, pointcloud_np):
        """
        对点云进行体素降采样。
        """
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(pointcloud_np)
        downsampled_cloud = cloud_o3d.voxel_down_sample(self.voxel_size)
        downsampled_np = np.asarray(downsampled_cloud.points)
        return downsampled_np

    def pointcloud_process(self, pc_msg):
        """
        点云回调函数，处理接收到的点云数据。
        """
        # rospy.loginfo("接收到点云数据，开始处理...")
        cloud_points = list(point_cloud2.read_points(pc_msg, skip_nans=True, field_names=("x", "y", "z")))
        cloud_np = np.array(cloud_points, dtype=np.float32)

        # 应用外参矫正
        # cloud_np_corrected = self.apply_extrinsic_correction(cloud_np)

        # 对点云进行降采样
        downsampled_points = self.downsample_pointcloud(cloud_np)

        # 发布降采样后的点云
        self.publish_pointcloud(downsampled_points, frame_id="base_link")
        
        return downsampled_points