# 导入ROS相关模块
import rospy
from sensor_msgs.msg import Imu, PointCloud2, PointField

# 导入ICP和NDT配准模块
from ndt_registration import NDTRegistration
from icp_registration import ICPRegistration
from imu_pipline import ImuDataHandler  # 导入IMU数据处理模块
from preprocess import Preprocess
from gicp import GICPRegistration

# 导入其他必要的库
import os
import numpy as np
import traceback
import signal
import time
from scipy.spatial.transform import Rotation as R
import subprocess
import sys
import open3d as o3d
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

# 设置环境变量
os.environ['NUMEXPR_MAX_THREADS'] = '12'

def transform_points(points, transformation_matrix):
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


class PointCloudProcessor:
    def __init__(self):
        imu_addr = "tcp://192.168.0.213:5680"
        self.imu_handler = ImuDataHandler(imu_addr)
        self.preprocess = Preprocess()
        self.ndt_registration = NDTRegistration()
        self.icp_registration = ICPRegistration()
        self.gicp = GICPRegistration(threshold=0.02, max_iterations=2000)
        self.map_points = np.empty((0, 3))  # 初始化为空的点云地图
        self.base_to_map = np.eye(4)  # 初始化变换矩阵
        self.frame_count = 0
        self.pc_pub = rospy.Publisher("/map_pointcloud", PointCloud2, queue_size=10)
        self.pc_gicp_pub = rospy.Publisher("/Gicp_pointcloud", PointCloud2, queue_size=10)
    
    def create_translation_matrix(self, dx=0, dy=0, dz=0):
            """
            创建一个平移矩阵。
            :param dx: x轴方向的平移量，默认为0。
            :param dy: y轴方向的平移量，默认为0。
            :param dz: z轴方向的平移量，默认为1（向前平移1米）。
            :return: 4x4的平移矩阵。
            """
            translation_matrix = np.eye(4)  # 创建一个4x4的单位矩阵
            translation_matrix[:3, 3] = [dx, dy, dz]  # 设置平移向量
            return translation_matrix


        
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
    
    def publish_GICP_pointcloud(self, data, frame_id):
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
            self.pc_gicp_pub.publish(pc_msg) 
              
    def process_pointcloud(self, pc_msg):
        """
        处理点云消息,包括预处理、NDT配准和ICP精配准。
        :param pc_msg: 点云消息。
        """
        if len(self.imu_handler.initial_imu_data) < 10:
            rospy.loginfo("IMU数据尚未初始化完成,跳过当前帧。")
            return

        # 1) 点云预处理
        self.frame_count = self.frame_count+1
        pre_points = self.preprocess.pointcloud_process(pc_msg)
        self.publish_pointcloud(self.map_points,"map")
        
        if self.frame_count == 1:
            # 如果是第一帧，将当前点云存储到地图中
            self.map_points = np.vstack((self.map_points, pre_points))
            rospy.loginfo("第一帧点云已存储到地图中")
            return  # 结束当前循环，跳过后续步骤
        
        if self.map_points.size == 0:
            rospy.loginfo("地图点云为空，跳过当前帧。")
            return
        
        # 假设每次都是平移一米，查看新建的地图
        # 2) 点云坐标转换
        rotation_matrix = self.imu_handler.rotation_matrix_to_001
        pointcloud_data_to_001 = pre_points @ rotation_matrix

        # # 3) NDT配准
        # transformation_ndt = self.ndt_registration.perform_ndt_registration(pointcloud_data_to_001, self.map_points)

        # 4) ICP精配准
        # transformation_icp = self.icp_registration.register(pointcloud_data_to_001, self.map_points)
        # transformation_icp = self.icp_registration.register(pointcloud_data_to_001, self.map_points)
        # rospy.loginfo(f"ICP配准位移: {transformation_icp[:3, 3]}")


        # GICP测试
        transformation_icp = self.gicp.register(pointcloud_data_to_001, self.map_points)
        rospy.loginfo(f"GICP配准位移: {transformation_icp[:3, 3]}")
        
        # 更新变换矩阵和地图点云
        self.base_to_map = transformation_icp  @ self.base_to_map
        # self.base_to_map = transformation_icp @ transformation_ndt @ self.base_to_map
        self.imu_handler.publish_base_to_map_transform( self.base_to_map)
        
        pointcloud_data_to_001_to_map = transform_points(pointcloud_data_to_001, self.base_to_map)
        self.publish_GICP_pointcloud(pointcloud_data_to_001_to_map,"map")
        self.map_points = np.vstack((self.map_points, pointcloud_data_to_001_to_map))
        
        # 测试代码逻辑#######################################################
        # 创建向前平移1米的平移矩阵
        # translation_matrix = self.create_translation_matrix(dx=0.5)
        # print("平移矩阵：\n", translation_matrix)
        # # 更新变换矩阵和地图点云
        # self.base_to_map = translation_matrix @ self.base_to_map
        # self.imu_handler.publish_base_to_map_transform( self.base_to_map)
        # pointcloud_data_to_001_to_map = transform_points(pre_points, self.base_to_map)
        # self.map_points = np.vstack((self.map_points, pointcloud_data_to_001_to_map))
        # 测试代码逻辑#######################################################
        
        # 如果处理的帧数超过10帧，关闭程序
        if self.frame_count > 10:
            # rospy.loginfo("处理的帧数超过10帧，关闭程序")
            print("处理的帧数超过",self.frame_count,"帧，关闭程序")
            rospy.signal_shutdown("处理的帧数超过10帧,程序正常关闭")
            # 确保rosbag进程也被终止
            if hasattr(self, 'rosbag_process'):
                self.rosbag_process.terminate()
                self.rosbag_process.wait()
        else:
            print("正在进行第",self.frame_count,"帧的数据处理")
                
    def callback(self, pc_msg):
        """
        点云消息回调函数。
        :param pc_msg: 点云消息。
        """
        try:
            self.process_pointcloud(pc_msg)
        except Exception as e:
            rospy.logerr(f"处理点云时发生错误: {e}")
            rospy.logerr(traceback.format_exc())

    def start(self):
        """
        启动ROS节点并订阅点云话题。
        """
        # 注册点云订阅回调函数
        msg_name = "/sunny_topic/device_0A30_952B_10F9_3044/tof_frame/pointcloud_horizontal"
        rospy.Subscriber(msg_name, PointCloud2, self.callback)

        # 播放rosbag文件
        bag_file = os.path.expanduser("/home/gongyou/Documents/01_slam/icp-ndt-2.bag")
        with open("rosbag_output.log", "w") as log_file:
            rosbag_process = subprocess.Popen(
                ["rosbag", "play", "--clock", bag_file],
                stdout=log_file,
                stderr=log_file
            )
        time.sleep(2)

        rospy.spin()  # 保持节点运行，直到被关闭
        # rospy.loginfo("测试完成。")

        # 结束rosbag play进程
        rosbag_process.terminate()
        rosbag_process.wait()
        rospy.signal_shutdown("ROS节点结束。")


if __name__ == '__main__':
    
    rospy.init_node('pointclouds_imu', anonymous=True)

    processor = PointCloudProcessor()
    processor.start()