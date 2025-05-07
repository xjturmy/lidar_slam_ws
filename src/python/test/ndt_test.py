import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

class NDT:
    def __init__(self, reference_points, resolution=1.0):
        """
        初始化 NDT 类
        :param reference_points: 参考点云 (N, 3) 的 numpy 数组
        :param resolution: 网格分辨率
        """
        self.reference_points = reference_points
        self.resolution = resolution
        self.grid = self._create_grid(reference_points, resolution)

    def _create_grid(self, points, resolution):
        """
        创建网格并计算每个网格单元的均值和协方差
        :param points: 点云数据
        :param resolution: 网格分辨率
        :return: 网格字典，键为网格索引，值为网格信息（均值和协方差）
        """
        grid = {}
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        grid_size = np.ceil((max_bound - min_bound) / resolution).astype(int)

        for point in points:
            grid_index = ((point - min_bound) / resolution).astype(int)
            grid_index = tuple(grid_index)

            if grid_index not in grid:
                grid[grid_index] = {"points": []}

            grid[grid_index]["points"].append(point)

        for index in grid:
            points_in_cell = np.array(grid[index]["points"])
            mean = np.mean(points_in_cell, axis=0)

            if len(points_in_cell) > 1:
                cov = np.cov(points_in_cell, rowvar=False)
            else:
                cov = np.eye(3) * 0.01  # 默认协方差矩阵（小的正定矩阵）

            grid[index]["mean"] = mean
            grid[index]["cov"] = cov

        return grid

    def _evaluate(self, params, source_points):
        """
        评估 NDT 的目标函数
        :param params: 变换参数（平移和旋转）
        :param source_points: 源点云
        :return: 残差
        """
        tx, ty, tz, qx, qy, qz, qw = params
        translation = np.array([tx, ty, tz])
        rotation = R.from_quat([qx, qy, qz, qw])

        transformed_points = rotation.apply(source_points) + translation
        residuals = []

        for point in transformed_points:
            grid_index = ((point - np.min(self.reference_points, axis=0)) / self.resolution).astype(int)
            grid_index = tuple(grid_index)

            if grid_index in self.grid:
                mean = self.grid[grid_index]["mean"]
                cov = self.grid[grid_index]["cov"]

                # 添加正则化项
                cov += np.eye(3) * 1e-6

                # 使用伪逆
                inv_cov = np.linalg.pinv(cov)

                residual = np.dot((point - mean).T, np.dot(inv_cov, (point - mean)))
                residuals.append(residual)

        return np.array(residuals)

    def match(self, source_points, initial_guess=None):
        """
        使用 NDT 匹配源点云到参考点云
        :param source_points: 源点云
        :param initial_guess: 初始变换参数（平移和旋转四元数）
        :return: 最优变换参数（平移和旋转四元数）
        """
        if initial_guess is None:
            initial_guess = [0, 0, 0, 0, 0, 0, 1]  # 默认初始值为无变换

        result = least_squares(self._evaluate, initial_guess, args=(source_points,))
        return result.x

def quaternion_to_transformation_matrix(q):
    """
    将四元数转换为变换矩阵
    :param q: 四元数 [qx, qy, qz, qw]
    :return: 4x4 变换矩阵
    """
    r = R.from_quat(q)
    rotation_matrix = r.as_matrix()
    translation = np.array([q[0], q[1], q[2]])

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation

    return transformation_matrix

# 示例使用
if __name__ == "__main__":
    # 随机生成参考点云
    reference_points = np.random.rand(100, 3) * 10

    # 将参考点云在 x 轴上平移 0.1 米，生成源点云
    source_points = reference_points.copy()
    source_points[:, 0] += 0.5

    # 将 z 轴的值设置为 0
    reference_points[:, 2] = 0
    source_points[:, 2] = 0

    # 初始化 NDT 并进行匹配
    ndt = NDT(reference_points, resolution=1.0)
    initial_guess = [0, 0, 0, 0, 0, 0, 1]  # 初始变换参数
    result = ndt.match(source_points, initial_guess)

    # 输出结果
    print("最优变换参数（平移和旋转四元数）:", result)

    # 将四元数转换为变换矩阵
    transformation_matrix = quaternion_to_transformation_matrix(result[3:])
    print("变换矩阵：")
    print(transformation_matrix)