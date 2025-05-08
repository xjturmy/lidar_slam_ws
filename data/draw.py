import csv
import matplotlib.pyplot as plt

# 加载 CSV 文件
file_path = '/home/gongyou/Documents/01_slam/lidar_slam_ws/data/registration_distances.csv'

# 初始化数据列表
translations = []
icp_distances = []
ndt_distances = []
gicp_distances = []

# 读取 CSV 文件
with open(file_path, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        translations.append(float(row['位移']))
        icp_distances.append(float(row['ICP距离']))
        ndt_distances.append(float(row['NDT距离']))
        gicp_distances.append(float(row['GICP距离']))

# 计算每种配准方法的误差均值
icp_mean_distance = sum(icp_distances) / len(icp_distances)
ndt_mean_distance = sum(ndt_distances) / len(ndt_distances)
gicp_mean_distance = sum(gicp_distances) / len(gicp_distances)

# 在终端中输出误差的均值
print(f"ICP Mean Distance: {icp_mean_distance:.6f}")
print(f"NDT Mean Distance: {ndt_mean_distance:.6f}")
print(f"GICP Mean Distance: {gicp_mean_distance:.6f}")

# 绘制折线图
plt.figure(figsize=(10, 6))

# ICP 距离
plt.plot(translations, icp_distances, label='ICP Distance', marker='o', linestyle='-')

# NDT 距离
plt.plot(translations, ndt_distances, label='NDT Distance', marker='s', linestyle='-')

# GICP 距离
plt.plot(translations, gicp_distances, label='GICP Distance', marker='^', linestyle='-')

# 添加标题和标签
plt.title('Comparison of Correspondence Distances for Different Registration Methods')
plt.xlabel('Translation (m)')
plt.ylabel('Correspondence Distance')

# 添加图例
plt.legend()

# 添加网格
plt.grid(True)

# 显示图表
plt.show()