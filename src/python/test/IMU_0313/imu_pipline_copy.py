#!/usr/bin/env python
import rospy
import tf
import re
import zmq
import numpy as np
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R
from tf.transformations import quaternion_from_euler

# from base_data import Rigid3d, TimedPose, ImuData, TimestampedTransform,Time
import threading

class ImuDataHandler:
    def __init__(self, imu_addr):
        # self.imu_addr = imu_addr
        # self.context = zmq.Context()
        # self.socket_sub = self.context.socket(zmq.SUB)
        # print("Connecting to IMU data source at:", imu_addr)
        # self.socket_sub.connect(self.imu_addr)
        # self.socket_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        # self.extrapolator_ = None


        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)
        


        self.initial_imu_data=[]
        self.current_ros_time_ = None

        self.rotation_matrix_to_001=np.eye(3)
        
        self.rotation_matrix_to_world = np.array([
            [-0.70985336, -0.15872489, -0.68623218],
            [-0.15872489,  0.98526564, -0.06370261],
            [ 0.68623218,  0.06370261, -0.72458772]
        ])

        # 创建 tf 广播器
        self.tf_broadcaster = tf.TransformBroadcaster()

        # 初始化速度和位移
        self.velocity = np.zeros(3)  # [vx, vy, vz]
        self.displacement = np.zeros(3)  # [x, y, z]
        self.last_time = rospy.Time.now()

        self.flag1=False
        self.flag2=False


    def broadcast_transform(self, displacement,quaternion,link1):
        """
        发布link1到link2坐标系的变换关系
        """

        # 发布变换关系
        self.tf_broadcaster.sendTransform(
            displacement,  # 假设 IMU 和基座之间没有平移
            quaternion,  # 四元数
            rospy.Time.now(),
            link1,  # IMU 坐标系
            "odom"  # 机器人基座坐标系
        )

        # 打印变换信息（可选）
        # rospy.loginfo(f"IMU to Base Transform: Roll={self.current_roll:.2f}, Pitch={self.current_pitch:.2f}, Yaw={self.current_yaw:.2f}")

    # def parse_message(self, message):
    #     pattern = r"ax_earth: (-?\d+\.\d+), ay_earth: (-?\d+\.\d+), az_earth: (-?\d+\.\d+), gx_earth: (-?\d+\.\d+), gy_earth: (-?\d+\.\d+), gz_earth: (-?\d+\.\d+)"
    #     match = re.search(pattern, message)
    #     if match:
    #         ax, ay, az = float(match.group(1)), float(match.group(2)), float(match.group(3))
    #         gx, gy, gz = float(match.group(4)), float(match.group(5)), float(match.group(6))

    #         # 加速度和角速度转换到世界坐标系
    #         acceleration_imu = np.array([ax, ay, az])
    #         angular_velocity_imu = np.array([gx, gy, gz])
    #         acceleration_world = np.dot(self.rotation_matrix_to_world, acceleration_imu)
    #         angular_velocity_world = np.dot(self.rotation_matrix_to_world, angular_velocity_imu)

    #         # 阈值处理
    #         ax, ay, az = map(lambda x: 0.0 if abs(x) < 0.00001 else x, acceleration_world)
    #         gx, gy, gz = map(lambda x: 0.0 if abs(x) < 0.00001 else x, angular_velocity_world)
    #         # 这里修改重力
    #         az = -az
    #         return ax, ay, az, gx, gy, gz

    #     rospy.logwarn("Failed to parse IMU data, setting default values to 0.0")
    #     return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # def create_imu_message(self, adjusted_linear_acceleration ,frame_id="imu_link"):
    #     ax,ay, az, gx, gy, gz=adjusted_linear_acceleration
    #     imu_msg = Imu()
    #     imu_msg.header = rospy.Header()
    #     imu_msg.header.stamp = rospy.Time.now()
    #     imu_msg.header.frame_id = frame_id

    #     imu_msg.linear_acceleration.x = ax
    #     imu_msg.linear_acceleration.y = ay
    #     imu_msg.linear_acceleration.z = az

    #     imu_msg.angular_velocity.x = gx
    #     imu_msg.angular_velocity.y = gy
    #     imu_msg.angular_velocity.z = gz

    #     imu_msg.linear_acceleration_covariance = [0.0] * 9
    #     imu_msg.angular_velocity_covariance = [0.0] * 9

    #     return imu_msg

    # def start_imu_data_thread(self, imu_pub):
    #     """
    #     启动一个线程，持续接收 IMU 数据并进行处理。
    #     """
    #     self.imu_pub = imu_pub
    #     self.running = True  # 设置一个标志，用于控制线程的运行状态
    #     self.imu_data_thread = threading.Thread(target=self.publish_imu_data_thread)
    #     self.imu_data_thread.start()

    def process_former10_imu(self,ax,ay,az):

        rospy.loginfo("正在处理前十帧数据")
        init_imu_data_accel = np.array([ax, ay, az])

        #imu数据使用上一帧imu数据的旋转矩阵进行矫正
        init_imu_data_accel = init_imu_data_accel @ self.rotation_matrix_to_001
        #计算当前帧与001的旋转轴和旋转角
        init_imu_data_normalized=init_imu_data_accel / np.linalg.norm(init_imu_data_accel)
        rotation_axis = np.cross([0, 0, 1], init_imu_data_normalized)
        cos_theta = np.dot([0, 0, 1], init_imu_data_normalized)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        #旋转矩阵累积
        rotation_matrix = R.from_rotvec(theta * rotation_axis).as_matrix()
        self.rotation_matrix_to_001 = self.rotation_matrix_to_001 @ rotation_matrix

        self.initial_imu_data.append(init_imu_data_accel)
        return 

    def caculate_velocity_displacement(self, acceleration, dt):
        """
        根据加速度累积计算速度和位移
        """
        self.velocity += acceleration * dt
        self.displacement += self.velocity * dt


    # def publish_imu_data_thread(self):
    #     """
    #     线程函数，持续接收 IMU 数据并进行处理。
    #     """
    #     rate=rospy.Rate(10)
    #     while self.running and not rospy.is_shutdown():
    #         try:
    #             # rospy.loginfo("Attempting to receive IMU data...")
    #             message = self.socket_sub.recv_string(flags=zmq.NOBLOCK)  # 使用非阻塞方式接收数据
    #             ax, ay, az, gx, gy, gz = self.parse_message(message)




    #             if [ax, ay, az] == [0.0, 0.0, 0.0]:
    #                 continue
                
    #             if len(self.initial_imu_data) < 10:
    #                 self.process_former10_imu(ax,ay,az)
    #                 # 前十帧，直接返回
    #                 continue

    #             elif len(self.initial_imu_data) == 10:
    #                 rotation = R.from_matrix(self.rotation_matrix_to_001)
    #                 self.quaternion = rotation.as_quat()

    #             #发布imu——link到base_link之间的坐标转换
    #             self.broadcast_transform((0,0,0.1),self.quaternion,"imu_link")
    #             # 矫正加速度计数据
    #             corrected_accel_data = np.array([ax, ay, az]) @ self.rotation_matrix_to_001
    #             # 矫正陀螺仪数据
    #             corrected_gyro_data = np.array([gx, gy, gz]) @ self.rotation_matrix_to_001 
                
    #             #计算位移量
    #             current_time = rospy.Time.now()
    #             dt = (current_time - self.last_time).to_sec()
    #             self.last_time = current_time
    #             self.caculate_velocity_displacement(corrected_accel_data,dt)
    #             if 0.5<self.displacement[0]<0.6 or self.flag1==True: 
    #                 self.broadcast_transform([0.5,0,0],quaternion_from_euler(0, 0, 0),"0.5米位置")
    #                 self.flag1=True
    #             if 1.0<self.displacement[0]<1.2 or self.flag2==True:
    #                 self.broadcast_transform([1,0,0],quaternion_from_euler(0, 0, 0),"1米位置")
    #                 self.flag2=True
    #             # print("实时速度")
    #             # print(self.velocity)
    #             # print("实时位移")
    #             # print(self.displacement)


    #             # print("旋转矩阵\n",self.rotation_matrix_to_001)
    #             # print("10帧之后的原始数据\n",[ax, ay, az, gx, gy, gz])
    #             # print("加速度\n",corrected_accel_data)
    #             # print("下一帧\n")

   
    #             # imu_msg = self.create_imu_message(adjusted_linear_acceleration)
    #             # self.imu_pub.publish(imu_msg)
                
    #             # rospy.loginfo("Published IMU data...")

    #         except zmq.Again:
    #             rospy.loginfo("No IMU data available.")
    #             self.last_time=rospy.Time.now()
    #         except Exception as e:
    #             rospy.logerr("Error processing IMU data: %s", str(e))
    #             self.last_time=rospy.Time.now()
    #         rate.sleep()  # 控制循环频率

    def imu_callback(self, imu_msg):
        """
        IMU数据回调函数
        """
        try:
            ax = imu_msg.linear_acceleration.x
            ay = imu_msg.linear_acceleration.y
            az = imu_msg.linear_acceleration.z
            gx = imu_msg.angular_velocity.x
            gy = imu_msg.angular_velocity.y
            gz = imu_msg.angular_velocity.z

            if [ax, ay, az] == [0.0, 0.0, 0.0]:
                return
                
            if len(self.initial_imu_data) < 10:
                self.process_former10_imu(ax,ay,az)
                # 前十帧，直接返回
                return

            elif len(self.initial_imu_data) == 10:
                rotation = R.from_matrix(self.rotation_matrix_to_001)
                self.quaternion = rotation.as_quat()

            #发布imu——link到base_link之间的坐标转换
            # self.broadcast_transform((0,0,0),self.quaternion,"起点")
            # self.broadcast_transform((2,0,0),self.quaternion,"base_link")

            # 矫正加速度计数据
            corrected_accel_data = np.array([ax, ay, az]) @ self.rotation_matrix_to_001
            # 矫正陀螺仪数据
            corrected_gyro_data = np.array([gx, gy, gz]) @ self.rotation_matrix_to_001 
            
            #计算位移量
            current_time = rospy.Time.now()
            dt = (current_time - self.last_time).to_sec()
            self.last_time = current_time
            self.caculate_velocity_displacement(corrected_accel_data,dt)

            # self.broadcast_transform((0,0,0),self.quaternion,"base_link")
            # if 0.5<self.displacement[0]<0.6 or self.flag1==True: 
            #     self.broadcast_transform([0.5,0,0],quaternion_from_euler(0, 0, 0),"0.5米位置")
            #     self.flag1=True
            # if 1.0<self.displacement[0]<1.2 or self.flag2==True:
            #     self.broadcast_transform([1,0,0],quaternion_from_euler(0, 0, 0),"1米位置")
            #     self.flag2=True
            # print("实时速度")
            # print(self.velocity)
            # print("实时位移")
            # print(self.displacement)


            # print("旋转矩阵\n",self.rotation_matrix_to_001)
            # print("10帧之后的原始数据\n",[ax, ay, az, gx, gy, gz])
            # print("加速度\n",corrected_accel_data)
            # print("下一帧\n")


            # imu_msg = self.create_imu_message(adjusted_linear_acceleration)
            # self.imu_pub.publish(imu_msg)
            
            # rospy.loginfo("Published IMU data...")

        except zmq.Again:
            rospy.loginfo("No IMU data available.")
            self.last_time=rospy.Time.now()
        except Exception as e:
            rospy.logerr("Error processing IMU data: %s", str(e))
            self.last_time=rospy.Time.now()




    # def stop_imu_data_thread(self):
    #     """
    #     停止 IMU 数据接收线程。
    #     """
    #     self.running = False  # 设置标志为 False，停止线程
    #     self.imu_data_thread.join()  # 等待线程结束