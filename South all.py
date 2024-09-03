import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 定义函数：为每个地点的数据集添加标识列并生成滞后特征
def process_location_data(file_path, location_label):
    df = pd.read_csv(file_path)
    df['Location'] = location_label  # 添加地点标识
    df = df.dropna()  # 移除缺失值

    # 生成滞后特征
    weather_features = df.columns.difference(['ZeroTimes', 'ZeroTimes_Class', 'City', 'Date', 'Location'])
    for feature in weather_features:
        df[f'{feature}_lag1'] = df[feature].shift(1)
        df[f'{feature}_lag2'] = df[feature].shift(2)

    # 删除没有完整滞后特征的行
    df = df.dropna()

    # 将 ZeroTimes 转换为二元变量
    df['ZeroTimes_Class'] = df['ZeroTimes'].apply(lambda x: 0 if x == 0 else 1)

    return df

# 加载并处理四个地点的数据
locations = {
    'Chennai': 'C:/Users/shang/Desktop/IC/Sem3/All/chennai_all_data NO1419.csv',
    'Bel': 'C:/Users/shang/Desktop/IC/Sem3/All/Belweause.csv',
    'Beng': 'C:/Users/shang/Desktop/IC/Sem3/All/Bengwea.csv',
    'Cudd': 'C:/Users/shang/Desktop/IC/Sem3/All/Cuddwea.csv'
}

processed_dfs = [process_location_data(path, label) for label, path in locations.items()]
combined_df = pd.concat(processed_dfs, ignore_index=True)

# 打印每个地点的 ZeroTimes_Class 类别分布和数据数量
for location_label in locations.keys():
    location_data = combined_df[combined_df['Location'] == location_label]
    class_distribution = location_data['ZeroTimes_Class'].value_counts(normalize=True) * 100
    data_count = len(location_data)
    print(f"Class distribution for Location {location_label} (Total data points: {data_count}):")
    print(class_distribution)
    print("\n")

# 提取特征和目标变量，同时保留Location列
features = combined_df.drop(columns=['ZeroTimes', 'ZeroTimes_Class', 'City', 'Date'])
target_classes = combined_df['ZeroTimes_Class']

# 划分训练集和测试集（按地点分开），并保留 Location 列
X_train, X_test, y_train, y_test = train_test_split(features, target_classes, test_size=0.2, random_state=42,
                                                    stratify=combined_df['Location'])

# 分离 Location 列用于后续分割
X_train_location = X_train['Location']
X_test_location = X_test['Location']

# 删除 Location 列用于模型训练
X_train = X_train.drop(columns=['Location'])
X_test = X_test.drop(columns=['Location'])

# 使用SMOTE对训练集进行过采样
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 初始化随机森林模型
rf_model = RandomForestClassifier(random_state=42)

# 添加五折交叉验证
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train_resampled, y_train_resampled, cv=kfold, scoring='accuracy')

# 打印交叉验证结果
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()}")
print(f"Standard Deviation of CV Accuracy: {cv_scores.std()}")

# 训练模型
rf_model.fit(X_train_resampled, y_train_resampled)

# 分别测试每个地点的测试集
for location_label in locations.keys():
    test_data = X_test[X_test_location == location_label]
    y_test_loc = y_test[X_test_location == location_label]

    # 预测
    y_pred_loc = rf_model.predict(test_data)

    # 评估模型
    accuracy_loc = accuracy_score(y_test_loc, y_pred_loc)
    conf_matrix_loc = confusion_matrix(y_test_loc, y_pred_loc)
    class_report_loc = classification_report(y_test_loc, y_pred_loc)

    print(f"Results for Location {location_label}:")
    print(f"Accuracy: {accuracy_loc}")
    print("Confusion Matrix:")
    print(conf_matrix_loc)
    print("Classification Report:")
    print(class_report_loc)
    print("\n")

# 特征重要性
importances = rf_model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.sort_values(by='Importance', ascending=False))
plt.title('Feature Importance')
plt.show()
