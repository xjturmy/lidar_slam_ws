from scipy.optimize import minimize,least_squares
import numpy as np
from scipy.spatial import KDTree
from joblib import Parallel, delayed


class NDT:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def _compute_grid_indices(self, points):
        """计算网格索引"""
        return np.floor(points / self.grid_size).astype(int)

    def build_target_grid(self, target_points):
        target_grid = {}
        indices = self._compute_grid_indices(target_points)
        for i, point in enumerate(target_points):
            idx = tuple(indices[i])
            if idx not in target_grid:
                target_grid[idx] = []
            target_grid[idx].append(point)

        # 预计算协方差矩阵的逆和行列式
        for idx in list(target_grid.keys()):
            points = np.array(target_grid[idx])
            if len(points) >= 3:
                mean = np.mean(points, axis=0)
                cov = np.cov(points, rowvar=False)
                try:
                    inv_cov = np.linalg.pinv(cov)
                    det_cov = np.linalg.det(cov)
                    # 这里将元组赋值给 target_grid[idx]
                    target_grid[idx] = (mean, inv_cov, det_cov)
                except np.linalg.LinAlgError:
                    del target_grid[idx]
            else:
                del target_grid[idx]
        return target_grid

    def _ndt_cost_function(self, transform, source_points,target_grid):
        transform = transform.reshape(4, 4)
        transformed_points = np.dot(source_points, transform[:3, :3].T) + transform[:3, 3]
        indices = self._compute_grid_indices(transformed_points)

        valid_mask = np.array([tuple(idx) in target_grid and target_grid[tuple(idx)] is not None for idx in indices])
        valid_points = transformed_points[valid_mask]
        valid_indices = indices[valid_mask]

        costs = []
        for i, point in enumerate(valid_points):
            idx = tuple(valid_indices[i])
            mean, inv_cov, det_cov = target_grid[idx]
            diff = point - mean
            if det_cov > 1e-6:
                cost = np.log(det_cov) + np.dot(diff.T, inv_cov.dot(diff))
            else:
                cost = np.dot(diff.T, inv_cov.dot(diff))
            costs.append(cost)

        return np.sum(costs) if costs else 0.0

    def register(self, source_points, target_points, initial_transform=None):
        """执行NDT配准"""
        # 构建目标网格
        target_grid=self.build_target_grid(target_points)
        # print("tf register ndt is :", target_points)
        # 初始化变换矩阵
        if initial_transform is None:
            initial_transform = np.eye(4)
            # # 或者使用一个小的随机扰动
            # initial_transform[:3, 3] = np.random.normal(0, 0.1, 3)

        # 优化配置
        options = {
            'maxiter': 300,
            'disp': True,
            # 'tol': 1e-6
            
        }
        # 优化
        bounds = [(None, None)] * 16  # 默认无界
        # 限制平移部分的范围
        for i in range(12, 15):  # 对应平移部分（T.x, T.y, T.z）
            # bounds[i] = (-0.5, 0.5)
            bounds[i] = (-1, 1)

        result = minimize(
            fun=self._ndt_cost_function,
            x0=initial_transform.ravel(),
            method='Nelder-Mead',
            bounds=bounds,
            options=options,
            args=(source_points,target_grid)
        )
        # result = least_squares(fun=self._ndt_cost_function,x0=initial_transform.ravel(),args=(source_points,target_grid))
        
        return result.x.reshape(4, 4),result.fun