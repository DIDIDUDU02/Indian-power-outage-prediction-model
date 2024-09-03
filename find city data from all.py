import pandas as pd

# 第一步：读取地点信息文件
chennai_locations = pd.read_csv('/Users/shangtianze/Desktop/IC/Sem3/Processed data/filtered_chennai.csv')
location_names = chennai_locations['Location name'].unique()

# 第二步：读取大文件
# 修改路径到你的大文件的位置
large_file_path = '/path/to/large/file.csv'
large_data = pd.read_csv(large_file_path)


filtered_data = large_data[large_data['Location'].isin(location_names)]

# 第三步：追加数据到已存在的 CSV 文件中
# 修改路径到你的输出文件的位置
output_file_path = '/path/to/output/all_chennai.csv'
# 使用 mode='a' 来追加数据，header=False 表示不重复添加列名
filtered_data.to_csv(output_file_path, mode='a', header=False, index=False)
