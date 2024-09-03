import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
file_path = 'C:/Users/shang/Desktop/IC/Sem3/Chennai/chennai_all_data NO1419.csv'
data = pd.read_csv(file_path).dropna()

# 将日期列转换为日期类型，使用dayfirst参数
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# 筛选出停电时间大于600分钟和小于1分钟的数据
data_above_600 = data[data['ZeroTimes'] > 600]
data_below_200 = data[data['ZeroTimes'] < 1]

# 添加标签列用于标识数据集
data_above_600['Category'] = 'Above 600'
data_below_200['Category'] = 'No Outage'

# 合并数据集
combined_data = pd.concat([data_above_600, data_below_200])

# 设置Seaborn样式
sns.set(style="whitegrid")

# 绘制各特征的箱线图比较
plt.figure(figsize=(20, 18))

# 选择需要比较的特征列
features = ['Temperature Max', 'Temperature Min', 'Temperature', 'Dew Point', 'Humidity',
            'Precipitation', 'Precipitation Cover', 'Wind Gust', 'Wind Speed',
            'Pressure', 'Cloud Cover', 'Solar Radiation', 'Solar Energy']

for i, feature in enumerate(features, 1):
    plt.subplot(5, 3, i)
    sns.boxplot(x='Category', y=feature, data=combined_data)
    plt.title(f'Boxplot of {feature}')
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout(h_pad=2.5, pad=3.5)  # 增加顶部填充量，确保标题显示完整
plt.show()

# 绘制各特征的小提琴图比较
plt.figure(figsize=(20, 18))

for i, feature in enumerate(features, 1):
    plt.subplot(5, 3, i)
    sns.violinplot(x='Category', y=feature, data=combined_data)
    plt.title(f'Violin plot of {feature}')
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout(h_pad=5, pad= 3.5)  # 增加顶部填充量，确保标题显示完整
plt.show()

