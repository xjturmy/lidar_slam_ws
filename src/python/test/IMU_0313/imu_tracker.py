import numpy as np

class RotationMatrix:
    @classmethod
    def identity(cls):
        """返回单位旋转矩阵"""
        return np.eye(3)

    @classmethod
    def from_angular_velocity(cls, angular_velocity: np.ndarray, dt: float):
        """
        根据角速度和时间步长计算旋转矩阵。
        
        :param angular_velocity: 角速度向量
        :param dt: 时间步长
        :return: 旋转矩阵
        """
        theta = np.linalg.norm(angular_velocity) * dt
        if theta == 0:
            return cls.identity()
        
        axis = angular_velocity / np.linalg.norm(angular_velocity)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        rotation_matrix = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return rotation_matrix

    @staticmethod
    def inverse(rotation_matrix: np.ndarray):
        """返回旋转矩阵的逆"""
        return rotation_matrix.T

    @staticmethod
    def normalize(rotation_matrix: np.ndarray):
        """归一化旋转矩阵（确保其为正交矩阵）"""
        U, _, Vt = np.linalg.svd(rotation_matrix)
        return U @ Vt


class ImuTracker:
    def __init__(self):
        """
        初始化 IMU 跟踪器。

        :param imu_gravity_time_constant: 重力时间常数 (float，默认值为 10.0)
        """
        self.time_ = 0.0
        self.orientation_ = RotationMatrix.identity()  # 初始姿态为单位旋转矩阵
        self.gravity_vector_ = np.array([0.0, 0.0, 9.81])  # 假设重力方向为 Z 轴
        self.imu_angular_velocity_ = np.array([0.0, 0.0, 0.0])  # 初始角速度
        self.last_linear_acceleration_time_ = 0.0
        self.imu_gravity_time_constant_ = 10.0

    def Advance(self, time: float):
        """
        根据时间推进 IMU 跟踪器。

        :param time: 目标时间戳 (float)
        """
        assert self.time_ <= time, "Time must be greater than or equal to the current time"
        delta_t = time - self.time_

        # 使用旋转矩阵计算旋转
        rotation = RotationMatrix.from_angular_velocity(self.imu_angular_velocity_, dt=delta_t)

        # 更新朝向
        self.orientation_ = self.orientation_ @ rotation
        self.orientation_ = RotationMatrix.normalize(self.orientation_)  # 归一化旋转矩阵

        # 更新重力向量
        self.gravity_vector_ = RotationMatrix.inverse(rotation) @ self.gravity_vector_

        # 更新时间
        self.time_ = time

    def AddImuLinearAccelerationObservation(self, imu_linear_acceleration: np.ndarray):
        """
        添加线性加速度观测值。

        :param imu_linear_acceleration: 线性加速度 (numpy 数组)
        """
        delta_t = self.time_ - self.last_linear_acceleration_time_ if self.last_linear_acceleration_time_ > 0 else np.inf
        self.last_linear_acceleration_time_ = self.time_

        alpha = 1.0 - np.exp(-delta_t / self.imu_gravity_time_constant_)
        print(f"IMU data AddImuLinearAccelerationObservation imu_gravity_time_constant_ {self.imu_gravity_time_constant_}")
        self.gravity_vector_ = (1.0 - alpha) * self.gravity_vector_ + alpha * imu_linear_acceleration

        # 确保重力向量的 Z 分量为正
        assert self.gravity_vector_[2] > 0, "Gravity vector Z component must be positive"
        assert self.gravity_vector_[2] / np.linalg.norm(self.gravity_vector_) > 0.99, "Normalized gravity vector Z component must be greater than 0.99"

    def AddImuAngularVelocityObservation(self, imu_angular_velocity: np.ndarray):
        """
        添加角速度观测值。

        :param imu_angular_velocity: 角速度 (numpy 数组)
        """
        self.imu_angular_velocity_ = imu_angular_velocity