import numpy as np

class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        """初始化四元数，格式为 [w, x, y, z]"""
        self.quat = np.array([w, x, y, z], dtype=float)
        self.normalize_in_place()  # 确保四元数是单位四元数

    @classmethod
    def identity(cls):
        """返回单位四元数"""
        return cls(w=1.0, x=0.0, y=0.0, z=0.0)

    @classmethod
    def from_numpy(cls, quat):
        """从 numpy 数组创建四元数"""
        return cls(w=quat[0], x=quat[1], y=quat[2], z=quat[3])

    @classmethod
    def from_rotation(cls, rotation_matrix: np.ndarray):
        """
        从旋转矩阵创建四元数。
        
        :param rotation_matrix: 3x3 旋转矩阵
        :return: 对应的四元数
        """
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
                w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                x = 0.25 * s
                y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
                w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                y = 0.25 * s
                z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
                w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                z = 0.25 * s

        return cls(w=w, x=x, y=y, z=z)

    def normalize_in_place(self):
        """归一化四元数"""
        norm = np.linalg.norm(self.quat)
        if norm > 0:
            self.quat /= norm

    def normalized(self):
        """返回归一化后的四元数"""
        norm = np.linalg.norm(self.quat)
        if norm > 0:
            return Quaternion.from_numpy(self.quat / norm)
        return self

    def inverse(self):
        """返回四元数的逆"""
        w, x, y, z = self.quat
        return Quaternion(w=w, x=-x, y=-y, z=-z)

    def conjugate(self):
        """返回四元数的共轭"""
        w, x, y, z = self.quat
        return Quaternion(w=w, x=-x, y=-y, z=-z)

    def __mul__(self, other):
        """四元数乘法"""
        w1, x1, y1, z1 = self.quat
        w2, x2, y2, z2 = other.quat
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return Quaternion(w=w, x=x, y=y, z=z)

    def as_numpy(self):
        """返回四元数的 numpy 表示"""
        return self.quat

    @classmethod
    def exp(cls, angular_velocity, dt=1.0):
        """将角速度转换为四元数表示的旋转"""
        angle = np.linalg.norm(angular_velocity) * dt
        if angle == 0:
            return cls.identity()
        axis = angular_velocity / np.linalg.norm(angular_velocity)
        w = np.cos(angle / 2)
        x, y, z = np.sin(angle / 2) * axis
        return cls(w=w, x=x, y=y, z=z)

    def __repr__(self):
        return f"Quaternion(w={self.quat[0]}, x={self.quat[1]}, y={self.quat[2]}, z={self.quat[3]})"