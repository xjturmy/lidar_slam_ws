#!/usr/bin/env python
import rospy
import tf
import numpy as np
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R
from tf.transformations import quaternion_from_euler

class ImuDataHandler:
    def __init__(self, imu_addr):
        """
        IMU数据处理类初始化。
        :param imu_addr: IMU数据源地址。
        """
        # 初始化ROS订阅器
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.tf_broadcaster = tf.TransformBroadcaster()  # TF广播器
        # 初始化变量
        self.initial_imu_data = []
        self.rotation_matrix_to_001 = np.eye(3)
        self.rotation_matrix_to_world = np.array([
            [-0.70985336, -0.15872489, -0.68623218],
            [-0.15872489,  0.98526564, -0.06370261],
            [ 0.68623218,  0.06370261, -0.72458772]
        ])
        self.velocity = np.zeros(3)  # 速度 [vx, vy, vz]
        self.displacement = np.zeros(3)  # 位移 [x, y, z]
        self.last_time = rospy.Time.now()  # 上一次时间戳
        
        self.tf_broadcaster = tf.TransformBroadcaster()  # TF广播器
        self.flag1 = False
        self.flag2 = False
                      
    def publish_base_to_map_transform(self,base_to_map):
        """
        发布 base_link 到 map 的变换关系。
        """
        # 假设 base_link 到 map 的变换矩阵是 self.base_to_map
        translation = base_to_map[:3, 3]  # 提取平移部分
        rotation = R.from_matrix(base_to_map[:3, :3]).as_quat()  # 提取旋转部分并转换为四元数
        self.tf_broadcaster.sendTransform(
            translation,
            rotation,
            rospy.Time.now(),
            "base_link",
            "map"
        )
    
    def process_former10_imu(self, ax, ay, az):
        """
        处理前10帧IMU数据。
        :param ax: 加速度计x分量。
        :param ay: 加速度计y分量。
        :param az: 加速度计z分量。
        """
        rospy.loginfo("正在处理前十帧数据")
        init_imu_data_accel = np.array([ax, ay, az])
        init_imu_data_accel = init_imu_data_accel @ self.rotation_matrix_to_001
        init_imu_data_normalized = init_imu_data_accel / np.linalg.norm(init_imu_data_accel)
        rotation_axis = np.cross([0, 0, 1], init_imu_data_normalized)
        cos_theta = np.dot([0, 0, 1], init_imu_data_normalized)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        rotation_matrix = R.from_rotvec(theta * rotation_axis).as_matrix()
        self.rotation_matrix_to_001 = self.rotation_matrix_to_001 @ rotation_matrix
        self.initial_imu_data.append(init_imu_data_accel)

    def caculate_velocity_displacement(self, acceleration, dt):
        """
        根据加速度累积计算速度和位移。
        :param acceleration: 加速度向量。
        :param dt: 时间间隔。
        """
        self.velocity += acceleration * dt
        self.displacement += self.velocity * dt

    def imu_callback(self, imu_msg):
        """
        IMU数据回调函数。
        :param imu_msg: IMU消息。
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
                self.process_former10_imu(ax, ay, az)
                return

            elif len(self.initial_imu_data) == 10:
                rotation = R.from_matrix(self.rotation_matrix_to_001)
                self.quaternion = rotation.as_quat()

            corrected_accel_data = np.array([ax, ay, az]) @ self.rotation_matrix_to_001
            corrected_gyro_data = np.array([gx, gy, gz]) @ self.rotation_matrix_to_001

            current_time = rospy.Time.now()
            dt = (current_time - self.last_time).to_sec()
            self.last_time = current_time
            self.caculate_velocity_displacement(corrected_accel_data, dt)

        except Exception as e:
            rospy.logerr("Error processing IMU data: %s", str(e))
            self.last_time = rospy.Time.now()

if __name__ == '__main__':
    rospy.init_node('imu_data_handler', anonymous=True)
    imu_addr = "tcp://192.168.0.213:5680"
    imu_handler = ImuDataHandler(imu_addr)
    rospy.spin()