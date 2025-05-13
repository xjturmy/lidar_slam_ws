import pandas as pd
import matplotlib.pyplot as plt

# 读取三个CSV文件
df_current = pd.read_csv('statistics_Current_Last.csv')  # 当前帧数据
df_nicp = pd.read_csv('statistics_NDT_ICP.csv')  # NDT+ICP 数据
df_icp = pd.read_csv('statistics_ICP.csv')      # ICP 数据
df_ndt = pd.read_csv('statistics_NDT.csv')      # NDT 数据


# 只保留前200帧的数据
df_current = df_current.head(100)
df_nicp = df_nicp.head(100)
df_icp = df_icp.head(100)
df_ndt = df_ndt.head(100)
# 将数据单位从米转换为厘米

df_current['Distance'] *= 100
df_current['Mean'] *= 100
df_current['StdDev'] *= 100

df_nicp['Distance'] *= 100
df_nicp['Mean'] *= 100
df_nicp['StdDev'] *= 100

df_icp['Distance'] *= 100
df_icp['Mean'] *= 100
df_icp['StdDev'] *= 100

df_ndt['Distance'] *= 100
df_ndt['Mean'] *= 100
df_ndt['StdDev'] *= 100

# 绘制图表
plt.figure(figsize=(18, 12))

# 第一个子图：距离
plt.subplot(3, 1, 1)
plt.plot(df_current['Frame'], df_current['Distance'], label='Current Distance', marker='.')
plt.plot(df_nicp['Frame'], df_nicp['Distance'], label='NDT+ICP Distance', marker='o')
plt.plot(df_icp['Frame'], df_icp['Distance'], label='ICP Distance', marker='.')
plt.plot(df_ndt['Frame'], df_ndt['Distance'], label='NDT Distance', marker='.')
plt.xlabel('Frame')
plt.ylabel('Distance (cm)')
plt.title('Distance Comparison')
plt.legend()

# 第二个子图：平均值
plt.subplot(3, 1, 2)
plt.plot(df_current['Frame'], df_current['Mean'], label='Current Mean', marker='.')
plt.plot(df_nicp['Frame'], df_nicp['Mean'], label='NDT+ICP Mean', marker='o')
plt.plot(df_icp['Frame'], df_icp['Mean'], label='ICP Mean', marker='.')
plt.plot(df_ndt['Frame'], df_ndt['Mean'], label='NDT Mean', marker='.')
plt.xlabel('Frame')
plt.ylabel('Mean (cm)')
plt.title('Mean Comparison')
plt.legend()

# 第三个子图：标准差
plt.subplot(3, 1, 3)
plt.plot(df_current['Frame'], df_current['StdDev'], label='Current StdDev', marker='.')
plt.plot(df_nicp['Frame'], df_nicp['StdDev'], label='NDT+ICP StdDev', marker='o')
plt.plot(df_icp['Frame'], df_icp['StdDev'], label='ICP StdDev', marker='.')
plt.plot(df_ndt['Frame'], df_ndt['StdDev'], label='NDT StdDev', marker='.')
plt.xlabel('Frame')
plt.ylabel('StdDev (cm)')
plt.title('StdDev Comparison')
plt.legend()

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.show()