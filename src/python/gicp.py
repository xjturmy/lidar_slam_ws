import numpy as np
import open3d as o3d

class GICPRegistration:
    def __init__(self, threshold=0.02, max_iterations=2000):
        """
        初始化 GICP 配准类。
        :param threshold: 对应点对的最大距离阈值。
        :param max_iterations: 最大迭代次数。
        """
        self.threshold = threshold
        self.max_iterations = max_iterations

    def preprocess_point_cloud(self, pcd, voxel_size):
        """
        预处理点云，包括下采样和法线估计。
        :param pcd: 输入点云。
        :param voxel_size: 下采样体素大小。
        :return: 预处理后的点云。
        """
        pcd_down = pcd.voxel_down_sample(voxel_size)
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        return pcd_down

    def execute_gicp(self, source, target, trans_init):
        """
        执行 GICP 算法。
        :param source: 源点云。
        :param target: 目标点云。
        :param trans_init: 初始变换矩阵。
        :return: GICP 算法的结果。
        """
        # 确保法线已经计算
        if not source.has_normals():
            source.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.threshold * 2, max_nn=30))
        if not target.has_normals():
            target.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.threshold * 2, max_nn=30))

        gicp_result = o3d.pipelines.registration.registration_generalized_icp(
            source, target, self.threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations)
        )
        return gicp_result

    def register(self, current_points_in_map, map_points,trans_init, voxel_size=0.05):
        """
        执行 GICP 精配准并返回变换矩阵。
        :param current_points_in_map: 当前帧点云。
        :param map_points: 地图点云。
        :param voxel_size: 下采样体素大小。
        :return: GICP 算法的变换矩阵。
        """
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(current_points_in_map)
        source = self.preprocess_point_cloud(source, voxel_size)

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(map_points)
        target = self.preprocess_point_cloud(target, voxel_size)

        # trans_init = np.eye(4)
        gicp_result = self.execute_gicp(source, target, trans_init)
        transformation_gicp = gicp_result.transformation
        return transformation_gicp

# # 示例用法
# if __name__ == "__main__":
#     # 创建 GICP 配准实例
#     gicp = GICPRegistration(threshold=0.02, max_iterations=2000)

#     # 示例点云数据（替换为实际数据）
#     current_points_in_map = np.random.rand(100, 3)  # 当前帧点云
#     map_points = np.random.rand(100, 3)  # 地图点云

#     # 执行 GICP 配准
#     transformation = gicp.register(current_points_in_map, map_points, voxel_size=0.05)

#     print("GICP Transformation Matrix:\n", transformation)