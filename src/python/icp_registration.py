import numpy as np
import open3d as o3d

class ICPRegistration:
    def __init__(self):
        """
        初始化 ICP 配准类。
        :param threshold: 对应点对的最大距离阈值。
        :param max_iterations: 最大迭代次数。
        """
        threshold=0.02
        max_iterations=2000
        self.threshold = threshold
        self.max_iterations = max_iterations

    def execute_icp(self, source, target, trans_init):
        """
        执行 ICP 算法。
        :param source: 源点云。
        :param target: 目标点云。
        :param trans_init: 初始变换矩阵。
        :return: ICP 算法的结果。
        """
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target, self.threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations)
        )
        return icp_result

    def register(self, current_points_in_map, map_points):
        """
        执行 ICP 精配准并返回变换矩阵。
        :param current_points_in_map: 当前帧点云。
        :param map_points: 地图点云。
        :return: ICP 算法的变换矩阵。
        """
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(current_points_in_map)

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(map_points)

        trans_init = np.eye(4)
        icp_result = self.execute_icp(source, target, trans_init)
        transformation_icp = icp_result.transformation
        return transformation_icp