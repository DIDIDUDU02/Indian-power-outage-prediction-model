import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'C:/Users/shang/Desktop/IC/Sem3/Sit/Situse.csv'  # 请将这里的路径替换为您的文件路径
data = pd.read_csv(file_path)

# Extracting the 'ZeroTimes' column which represents the power outage minutes
zerotimes = data['ZeroTimes']

# Plotting the distribution of power outage minutes
plt.figure(figsize=(10, 6))
plt.hist(zerotimes, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Power Outage Minutes')
plt.xlabel('Power Outage Minutes')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
