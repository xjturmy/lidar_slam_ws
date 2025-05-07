import numpy as np
import open3d as o3d
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from preprocess import Preprocess
from NDT import NDT




# from NDT import NDT
class NDTRegistration:
    def __init__(self):
        self.input_topic = "/sunny_topic/device_0A30_952B_10F9_3044/tof_frame/pointcloud_horizontal"
        self.output_topic = "/ndt_registered_pointcloud"
        self.voxel_size =  0.05
        self.grid_size =  0.2
        # self.ndt = o3d.pipelines.registration.NDT()
        # self.ndt.set_resolution(self.grid_size)
        self.pc_pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=10)
        # pc_ndt_points_in_map = rospy.Publisher('/ndt_points_in_map', PointCloud2, queue_size=10)
        self.preprocess = Preprocess()

    def remove_ground_points_and_project_to_xy(self, points, z_threshold=0.1):
        """
        去除地面点并投影到 x-y 平面。
        """
        filtered_points = points[points[:, 2] >= z_threshold]
        filtered_points[:, 2] = 0
        return filtered_points

    def voxel_downsample(self, points):
        """
        对点云进行体素降采样。
        """
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(points)
        downsampled_cloud = cloud_o3d.voxel_down_sample(self.voxel_size)
        return np.asarray(downsampled_cloud.points)
    
    def transform_points(self,points, transformation_matrix):
        """
        将点云数据通过变换矩阵转换到新的坐标系。
        :param points: 输入点云，形状为 (N, 3) 的 NumPy 数组。
        :param transformation_matrix: 4x4 的变换矩阵。
        :return: 转换后的点云。
        """
        rotation = transformation_matrix[:3, :3]
        translation = transformation_matrix[:3, 3]
        transformed_points = points @ rotation.T + translation
        return transformed_points
    
    def perform_ndt_registration(self, source_points, target_points):
        """
        执行 NDT 粗配准并返回变换矩阵。
        """
        # source_cloud = o3d.geometry.PointCloud()
        # source_cloud.points = o3d.utility.Vector3dVector(source_points)

        # target_cloud = o3d.geometry.PointCloud()
        # target_cloud.points = o3d.utility.Vector3dVector(target_points)

        # self.ndt.set_input(source_cloud)
        # self.ndt.set_input(target_cloud)
         # 初始化 NDT 配准器
        ndt = NDT(grid_size=0.2)
        initial_transformation = np.eye(4)
        transformation_ndt, cost = ndt.register(source_points, target_points, initial_transformation)
        
        
                # 执行 NDT 粗配准
        # transformation_ndt = self.perform_ndt_registration(downsampled_points, downsampled_points)

        # 将点云数据通过变换矩阵转换到新的坐标系
        transformed_points = self.transform_points(source_points, transformation_ndt)

        # 发布变换后的点云
        self.preprocess.publish_pointcloud(transformed_points, frame_id="map")
        return transformation_ndt
        # transformation_ndt = self.ndt.compute_transformation()
        # return transformation_ndt

    def process_and_publish(self, pc_msg, base_to_map):
        """
        处理点云数据并发布结果。
        """
        cloud_points = list(point_cloud2.read_points(pc_msg, skip_nans=True, field_names=("x", "y", "z")))
        cloud_np = np.array(cloud_points, dtype=np.float32)

        # 去掉地面点并投影到 x-y 平面
        filtered_points = self.remove_ground_points_and_project_to_xy(cloud_np)

        # 体素降采样
        downsampled_points = self.voxel_downsample(filtered_points)

        # 执行 NDT 粗配准
        transformation_ndt = self.perform_ndt_registration(downsampled_points, downsampled_points)

        # 将点云数据通过变换矩阵转换到新的坐标系
        transformed_points = self.preprocess.transform_points(downsampled_points, transformation_ndt)

        # 发布变换后的点云
        self.preprocess.publish_pointcloud(transformed_points, frame_id=pc_msg.header.frame_id)
        return transformation_ndt

    def callback(self, pc_msg):
        print("当前测试点云的帧数")
        self.process_and_publish(pc_msg, np.eye(4))
