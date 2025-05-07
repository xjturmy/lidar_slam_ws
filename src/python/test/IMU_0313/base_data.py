# base_data.py
import numpy as np

class Rigid3d:
    def __init__(self, translation: np.ndarray, rotation: np.ndarray):
        self.translation = translation
        self.rotation = rotation

    @staticmethod
    def identity():
        """
        返回单位变换的 Rigid3d 对象。
        """
        translation = np.zeros(3)  # 平移部分为零向量
        rotation = np.eye(3)  # 旋转部分为单位矩阵
        return Rigid3d(translation, rotation)

    def __repr__(self):
        return (f"Rigid3d(translation={self.translation}, "
                f"rotation=\n{self.rotation})")

class Time:
    def __init__(self, timestamp: float):
        self.timestamp = timestamp

    @staticmethod
    def min():
        return Time(float('-inf'))  # 返回一个表示最小时间的对象

    def __repr__(self):
        return f"Time({self.timestamp})"

    def __lt__(self, other):
        if not isinstance(other, Time):
            raise TypeError("Comparison with non-Time object is not supported")
        return self.timestamp < other.timestamp

    def __le__(self, other):
        if not isinstance(other, Time):
            raise TypeError("Comparison with non-Time object is not supported")
        return self.timestamp <= other.timestamp

    def __sub__(self, other):
        if not isinstance(other, Time):
            raise TypeError("Subtraction with non-Time object is not supported")
        return self.timestamp - other.timestamp  # 返回时间差（浮点数）

class TimedPose:
    """
    表示带时间戳的姿态。
    """
    def __init__(self, time: 'Time', pose: 'Rigid3d'):
        """
        初始化 TimedPose 对象。

        :param time: 时间戳 (Time 对象)
        :param pose: 姿态 (Rigid3d 对象)
        """
        if not isinstance(time, Time):
            raise TypeError("Expected 'time' to be an instance of Time")
        if not isinstance(pose, Rigid3d):
            raise TypeError("Expected 'pose' to be an instance of Rigid3d")

        self.time = time
        self.pose = pose

    def __repr__(self):
        return f"TimedPose(time={self.time}, pose={self.pose})"

class ImuData:
    def __init__(self, time: Time, linear_acceleration: np.ndarray, angular_velocity: np.ndarray):
        if not isinstance(time, Time):
            raise TypeError("Expected 'time' to be an instance of Time")
        self.time = time
        self.linear_acceleration = linear_acceleration
        self.angular_velocity = angular_velocity
        
class TimestampedTransform:
    """
    带时间戳的变换类，包含时间戳和 3D 刚体变换。
    """
    time: float  # 时间戳，可以是秒级时间戳
    transform: np.ndarray  # 3D 刚体变换矩阵，通常是 4x4 的齐次变换矩阵



 