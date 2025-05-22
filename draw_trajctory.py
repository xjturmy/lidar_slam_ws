import evo
import evo.tools.file_interface as file_interface
import evo.core.trajectory as trajectory
import evo.core.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

# 加载轨迹文件
traj1 = file_interface.read_tum_trajectory_file("reality_trajectory.csv")
traj2 = file_interface.read_tum_trajectory_file("No_Optimization_icp.csv")
traj3 = file_interface.read_tum_trajectory_file("No_Optimization_gicp.csv")

# 将轨迹转换为3D位姿轨迹对象
traj1_3d = trajectory.PoseTrajectory3D(traj1)
traj2_3d = trajectory.PoseTrajectory3D(traj2)
traj3_3d = trajectory.PoseTrajectory3D(traj3)

# 计算ATE
ate_metric = metrics.ATE(traj1_3d, traj2_3d)
ate_result12 = ate_metric.compute()
ate_metric = metrics.ATE(traj1_3d, traj3_3d)
ate_result13 = ate_metric.compute()
ate_metric = metrics.ATE(traj2_3d, traj3_3d)
ate_result23 = ate_metric.compute()

# 打印ATE结果
print("ATE between Trajectory 1 and 2:", ate_result12)
print("ATE between Trajectory 1 and 3:", ate_result13)
print("ATE between Trajectory 2 and 3:", ate_result23)

# 绘制轨迹图
def plot_trajectories(trajectories, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for traj in trajectories:
        ax.plot(traj.positions[:, 0], traj.positions[:, 1], traj.positions[:, 2], label=f"Trajectory {traj}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()

# 绘制所有轨迹
plot_trajectories([traj1_3d, traj2_3d, traj3_3d], "Comparison of Three Trajectories")

# 如果需要，可以绘制误差图
# ate_metric.plot(ax, traj1_3d, traj2_3d)