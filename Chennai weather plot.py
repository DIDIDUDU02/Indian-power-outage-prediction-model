import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
file_path = 'C:/Users/shang/Desktop/IC/Sem3/Chennai/chennai_all_data NO1419.csv'
data = pd.read_csv(file_path)

# 选择要绘制箱线图的列
columns_to_plot = [
    'Temperature Max', 'Temperature Min', 'Temperature', 'Dew Point', 'Humidity',
    'Precipitation', 'Precipitation Cover', 'Wind Gust', 'Wind Speed',
    'Pressure', 'Cloud Cover', 'Solar Radiation', 'Solar Energy'
]

# 创建一个图形来绘制所有箱线图
plt.figure(figsize=(20, 20))

# 为每个选定的列创建单独的箱线图
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(4, 4, i)
    plt.boxplot(data[column].dropna())
    plt.title(column)
    plt.xticks([])  # 移除横坐标

plt.tight_layout(h_pad=2.5, pad=3)  # 增加上下间距，并整体下移
plt.show()
