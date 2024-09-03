import pandas as pd


# 定义函数：读取文件，二元化ZeroTimes列，并计算01分布
def calculate_class_distribution(file_path):
    # 读取文件
    df = pd.read_csv(file_path)

    # 将 ZeroTimes 列转换为二元变量
    df['ZeroTimes_Class'] = df['ZeroTimes'].apply(lambda x: 0 if x == 0 else 1)

    # 计算01分布
    class_distribution = df['ZeroTimes_Class'].value_counts(normalize=True) * 100

    return class_distribution


# 文件路径
file_paths = [
    'C:/Users/shang/Desktop/IC/Sem3/All/Bahr.csv',
    'C:/Users/shang/Desktop/IC/Sem3/All/Barause.csv',
    'C:/Users/shang/Desktop/IC/Sem3/All/Situse.csv'
]

# 计算并打印每个文件的01分布
for file_path in file_paths:
    distribution = calculate_class_distribution(file_path)
    print(f"Class distribution for {file_path.split('/')[-1]}:")
    print(distribution)
    print("\n")
