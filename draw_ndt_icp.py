import matplotlib.pyplot as plt
import numpy as np

# 加载第一个CSV文件的数据
data1 = np.loadtxt('F20_after_optimization_Left_registration_distances.csv', delimiter=',')
# 取前204帧的数据
data1 = data1[:204, :]

# 加载第二个CSV文件的数据
data2 = np.loadtxt('F20_after_optimization_Right_registration_distances.csv', delimiter=',')
# 取前204帧的数据
data2 = data2[:204, :]

# 加载第三个CSV文件的数据
data3 = np.loadtxt('No_optimization_registration_distances.csv', delimiter=',')
# 取前204帧的数据
data3 = data3[:204, :]

# 设置幕布大小（例如，宽度8英寸，高度6英寸）
plt.figure(figsize=(4, 8))

# 提取x和y坐标
x1 = data1[:, 0]
y1 = data1[:, 1]

x2 = data2[:, 0]
y2 = data2[:, 1]

x3 = data3[:, 0]
y3 = data3[:, 1]

# 绘制第一个轨迹
# plt.plot(-y1, x1, label='多次优化', color='blue')

# 绘制第二个轨迹
plt.plot(-y2, x2, label='optimization only once', color='red')

# 绘制第三个轨迹
plt.plot(-y3, x3, label='no optimization', color='green')

# 添加图例
plt.xlabel('Y')
plt.ylabel('X')
plt.title('Registration Distances')
plt.legend()

# 调整坐标轴范围
plt.xlim(-0.2, 0.2)
plt.ylim(0, 2)

# 显示图形
plt.show()