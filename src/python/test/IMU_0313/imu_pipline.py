#!/usr/bin/env python
import rospy
import tf
import re
import zmq
import numpy as np
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R


from .base_data import Rigid3d, TimedPose, ImuData, TimestampedTransform,Time
import threading

class ImuDataHandler:
    def __init__(self, imu_addr):
        self.imu_addr = imu_addr
        self.context = zmq.Context()
        self.socket_sub = self.context.socket(zmq.SUB)
        print("Connecting to IMU data source at:", imu_addr)
        self.socket_sub.connect(self.imu_addr)
        self.socket_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.extrapolator_ = None
        
        self.initial_imu_data=[]
        self.current_ros_time_ = None
        # 旋转矩阵定义
        self.rotation_matrix = np.array([
            [-0.70985336, -0.15872489, -0.68623218],
            [-0.15872489,  0.98526564, -0.06370261],
            [ 0.68623218,  0.06370261, -0.72458772]
        ])

        # 创建 tf 广播器
        self.tf_broadcaster = tf.TransformBroadcaster()
        # 初始化姿态变量
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.last_time = rospy.Time.now()

    def broadcast_transform(self):
        """
        发布从 IMU 坐标系到机器人基座坐标系的变换关系
        """

        # 将欧拉角转换为四元数
        # qx, qy, qz, qw = self.euler_to_quaternion(self.current_roll, self.current_pitch, self.current_yaw)

        # 发布变换关系
        self.tf_broadcaster.sendTransform(
            (0, 0, 0.1),  # 假设 IMU 和基座之间没有平移
            self.quaternion,  # 四元数
            rospy.Time.now(),
            "imu_link",  # IMU 坐标系
            "base_link"  # 机器人基座坐标系
        )

        # 打印变换信息（可选）
        rospy.loginfo(f"IMU to Base Transform: Roll={self.current_roll:.2f}, Pitch={self.current_pitch:.2f}, Yaw={self.current_yaw:.2f}")


    def parse_message(self, message):
        pattern = r"ax_earth: (-?\d+\.\d+), ay_earth: (-?\d+\.\d+), az_earth: (-?\d+\.\d+), gx_earth: (-?\d+\.\d+), gy_earth: (-?\d+\.\d+), gz_earth: (-?\d+\.\d+)"
        match = re.search(pattern, message)
        if match:
            ax, ay, az = float(match.group(1)), float(match.group(2)), float(match.group(3))
            gx, gy, gz = float(match.group(4)), float(match.group(5)), float(match.group(6))

            # 加速度和角速度转换到世界坐标系
            acceleration_imu = np.array([ax, ay, az])
            angular_velocity_imu = np.array([gx, gy, gz])
            acceleration_world = np.dot(self.rotation_matrix, acceleration_imu)
            angular_velocity_world = np.dot(self.rotation_matrix, angular_velocity_imu)

            # 阈值处理
            ax, ay, az = map(lambda x: 0.0 if abs(x) < 0.00001 else x, acceleration_world)
            gx, gy, gz = map(lambda x: 0.0 if abs(x) < 0.00001 else x, angular_velocity_world)
            # 这里修改重力
            az = -az
            return ax, ay, az, gx, gy, gz

        rospy.logwarn("Failed to parse IMU data, setting default values to 0.0")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0




    def create_imu_message(self, adjusted_linear_acceleration ,frame_id="imu_link"):
        ax,ay, az, gx, gy, gz=adjusted_linear_acceleration
        imu_msg = Imu()
        imu_msg.header = rospy.Header()
        imu_msg.header.stamp = rospy.Time.now()
        imu_msg.header.frame_id = frame_id

        imu_msg.linear_acceleration.x = ax
        imu_msg.linear_acceleration.y = ay
        imu_msg.linear_acceleration.z = az

        imu_msg.angular_velocity.x = gx
        imu_msg.angular_velocity.y = gy
        imu_msg.angular_velocity.z = gz

        imu_msg.linear_acceleration_covariance = [0.0] * 9
        imu_msg.angular_velocity_covariance = [0.0] * 9

        return imu_msg

    

    def start_imu_data_thread(self, imu_pub):
        """
        启动一个线程，持续接收 IMU 数据并进行处理。
        """
        self.imu_pub = imu_pub
        self.running = True  # 设置一个标志，用于控制线程的运行状态
        self.imu_data_thread = threading.Thread(target=self.publish_imu_data_thread)
        self.imu_data_thread.start()




    def publish_imu_data_thread(self):
        """
        线程函数，持续接收 IMU 数据并进行处理。
        """
        rate=rospy.Rate(20)

        while self.running and not rospy.is_shutdown():
            try:
                rospy.loginfo("Attempting to receive IMU data...")
                message = self.socket_sub.recv_string(flags=zmq.NOBLOCK)  # 使用非阻塞方式接收数据
                ax, ay, az, gx, gy, gz = self.parse_message(message)
                
                if len(self.initial_imu_data) < 15:
                    rospy.loginfo("正在收集前十五帧数据")
                    imu_data = ax, ay, az, gx, gy, gz
                    self.initial_imu_data.append(imu_data)
                    # 如果尚未收集到足够的数据，直接返回
                    continue
                elif len(self.initial_imu_data) == 15:
                    # 计算前十五帧IMU数据的加速度均值（去掉前4帧和后1帧，取中间10帧的均值）
                    initial_imu_data_mean = np.mean(self.initial_imu_data[4:14], axis=0)
                    # 归一化初始加速度向量
                    initial_imu_data_mean_normalized = initial_imu_data_mean[:3] / np.linalg.norm(initial_imu_data_mean[:3])
                    # 计算旋转轴
                    rotation_axis = np.cross([0, 0, 1], initial_imu_data_mean_normalized)
                    # 计算旋转角
                    cos_theta = np.dot([0, 0, 1], initial_imu_data_mean_normalized)
                    theta = np.arccos(cos_theta)
                    # 构造旋转矩阵
                    self.rotation_matrix = R.from_rotvec(theta * rotation_axis).as_matrix()
                    #将旋转矩阵转换为四元数
                    rotation = R.from_matrix(self.rotation_matrix)
                    self.quaternion = rotation.as_quat()

                #  accel_data 是后续的加速度计数据
                accel_data = np.array([ax, ay, az])  
                # 矫正加速度计数据
                corrected_accel_data = self.rotation_matrix @ accel_data
                #  gyro_data 是后续的陀螺仪数据
                gyro_data = np.array([gx, gy, gz])  
                # 矫正陀螺仪数据
                corrected_gyro_data = self.rotation_matrix @ gyro_data

                # gx, gy, gz=corrected_accel_data[0],corrected_accel_data[1],corrected_accel_data[2]
                # current_time = rospy.Time.now()
                # dt = (current_time - self.last_time).to_sec()
                # self.last_time = current_time
                # self.update_orientation(gx, gy, gz, dt)

                self.broadcast_transform()
                print("旋转矩阵\n",self.rotation_matrix)
                print("15帧之后的原始数据",accel_data,gyro_data)
                print("15帧之后imu数据矫正",corrected_accel_data,corrected_gyro_data)


   
                # imu_msg = self.create_imu_message(adjusted_linear_acceleration)
                # self.imu_pub.publish(imu_msg)
                
                # rospy.loginfo("Published IMU data...")

            except zmq.Again:
                rospy.loginfo("No IMU data available.")
            except Exception as e:
                rospy.logerr("Error processing IMU data: %s", str(e))

            rate.sleep()  # 控制循环频率

    def stop_imu_data_thread(self):
        """
        停止 IMU 数据接收线程。
        """
        self.running = False  # 设置标志为 False，停止线程
        self.imu_data_thread.join()  # 等待线程结束