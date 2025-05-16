import matplotlib.pyplot as plt
import numpy as np

# 加载轨迹数据
data = np.loadtxt('F20_after_optimization_Left_registration_distances.csv', delimiter=',')
#取前204帧的数据
data = data[:204, :]

# 设置幕布大小（例如，宽度8英寸，高度6英寸）
plt.figure(figsize=(4, 8))
# 提取x和y坐标
x = data[:, 0]
y = data[:, 1]

# 绘制轨迹
plt.plot(-y,x,label='registration distances')
plt.xlabel('Frame')
plt.ylabel('Distance')
plt.title('Registration Distances')
plt.legend()

# 调整坐标轴范围
plt.xlim(-0.2, 0.2)
plt.ylim(0, 2)

plt.show()