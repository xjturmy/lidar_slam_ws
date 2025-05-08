import csv
import matplotlib.pyplot as plt

# 加载 CSV 文件
file_path = '/home/gongyou/Documents/01_slam/lidar_slam_ws/data/ndt_distances.csv'

# 初始化数据列表
resolutions = []
ndt_distances = []

# 读取 CSV 文件
with open(file_path, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        resolutions.append(float(row['迭代次数']))
        ndt_distances.append(float(row['NDT距离']))

# 计算 NDT 距离的误差均值
ndt_mean_distance = sum(ndt_distances) / len(ndt_distances)

# 在终端中输出误差的均值
print(f"NDT Mean Distance: {ndt_mean_distance:.6f}")

# 绘制折线图
plt.figure(figsize=(10, 6))

# NDT 距离
plt.plot(resolutions, ndt_distances, label='NDT Distance', marker='o', linestyle='-')

# 添加标题和标签
plt.title('Comparison of NDT Correspondence Distances for Different Resolutions')
plt.xlabel('Resolution')
plt.ylabel('Correspondence Distance')

# 添加图例
plt.legend()

# 添加网格
plt.grid(True)

# 显示图表
plt.show()