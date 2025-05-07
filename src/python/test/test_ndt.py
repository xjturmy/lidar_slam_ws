import open3d as o3d
import numpy as np
import importlib

def check_open3d_version():
    """
    检查 Open3D 的版本。
    """
    version = o3d.__version__
    print(f"当前安装的 Open3D 版本: {version}")
    if version < "0.19.0":
        print("警告：当前 Open3D 版本低于 0.19.0，可能不支持 NDT 算法。")
    else:
        print("Open3D 版本符合要求，支持 NDT 算法。")

def check_module_availability(module_name):
    """
    检查模块是否可用。
    """
    try:
        module = importlib.import_module(module_name)
        print(f"模块 {module_name} 可用。")
        return module
    except ImportError as e:
        print(f"模块 {module_name} 不可用。错误信息：{e}")
        return None

def create_random_point_cloud(num_points=100, center=[0, 0, 0], scale=1.0):
    """
    创建一个随机点云。
    :param num_points: 点的数量。
    :param center: 点云的中心。
    :param scale: 点云的尺度。
    :return: 随机点云。
    """
    points = np.random.rand(num_points, 3) * scale + center
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def execute_ndt(source, target, voxel_size):
    """
    执行 NDT 算法。
    :param source: 源点云。
    :param target: 目标点云。
    :param voxel_size: 体素大小。
    :return: NDT 算法的结果。
    """
    try:
        # 预处理点云
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        # 初始化 NDT 配准器
        print("NDT test")
        ndt = o3d.pipelines.registration.registration_ndt.RegistrationNDT()

        # 设置NDT参数
        ndt.set_resolution(1.0)
        ndt.set_max_iterations(35)
        ndt.set_max_correspondence_distance(3.0)
        ndt.set_number_of_threads(4)

        # 进行配准
        reg_p2p = ndt.compute(source_down, target_down)

        return reg_p2p.transformation
    except AttributeError as e:
        print("NDT 算法不可用。错误信息：")
        print(e)
        return None

def main():
    print("Open3D 安装路径:", o3d.__file__)
    # 检查 Open3D 版本
    check_open3d_version()

    # 检查相关模块是否可用
    check_module_availability("open3d")
    check_module_availability("open3d.pipelines")
    # check_module_availability("open3d.pipelines.registration")

    # 创建两个随机点云
    source = create_random_point_cloud(num_points=100, center=[0, 0, 0], scale=1.0)
    target = create_random_point_cloud(num_points=100, center=[0.5, 0.5, 0.5], scale=1.0)

    # 设置体素大小
    voxel_size = 0.1

    # 执行 NDT 算法
    transformation_ndt = execute_ndt(source, target, voxel_size)
    if transformation_ndt is not None:
        print("NDT 算法的变换矩阵：")
        print(transformation_ndt)

        # 可视化结果
        source.transform(transformation_ndt)
        o3d.visualization.draw_geometries([source, target])

if __name__ == "__main__":
    main()