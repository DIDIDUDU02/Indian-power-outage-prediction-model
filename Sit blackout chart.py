import pandas as pd

# 加载数据
file_path = 'C:/Users/shang/Desktop/IC/Sem3/Sit/Situse.csv'
data = pd.read_csv(file_path)

# 假设停电时间列名为 'BlackoutMinutes'
# 如果列名不同，请更正
blackout_minutes = data['ZeroTimes']

# 定义时间间隔
bins = [0, 1, 20, 60, 120, 600, float('inf')]
labels = ['0 minute', '1-20 minutes', '21-60 minutes', '61-120 minutes', '121-600 minutes', '>600 minutes']

# 计算每个时间区间的分布
data['BlackoutMinutesInterval'] = pd.cut(blackout_minutes, bins=bins, labels=labels, right=False)

# 计算各个时间区间的计数和百分比
distribution = data['BlackoutMinutesInterval'].value_counts(sort=False)
percentage = distribution / distribution.sum() * 100

# 生成分布表格
distribution_df = pd.DataFrame({
    'Blackout Minutes Interval': labels,
    'Counts': distribution.values,
    'Percentage': percentage.round(2).astype(str) + '%'
})


print(distribution_df)

