import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
file_path = 'C:/Users/shang/Desktop/IC/Sem3/Chennai/chennai_all_data NO1419.csv'
alandur_process_df = pd.read_csv(file_path)

# 移除缺失值
cleaned_df = alandur_process_df.dropna()

# 将 ZeroTimes 转换为二元变量
cleaned_df['ZeroTimes_Class'] = cleaned_df['ZeroTimes'].apply(lambda x: 0 if x == 0 else 1)

# 提取特征和目标变量
features = cleaned_df.drop(columns=['ZeroTimes', 'ZeroTimes_Class', 'City', 'Date'])
target_classes = cleaned_df['ZeroTimes_Class']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target_classes, test_size=0.2, random_state=42)

# 使用默认参数创建随机森林模型
rf_model = RandomForestClassifier(random_state=42)

# 进行五折交叉验证
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kfold, scoring='accuracy')

# 打印交叉验证结果
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()}")
print(f"Standard Deviation of CV Accuracy: {cv_scores.std()}")

# 训练模型
rf_model.fit(X_train, y_train)

# 预测
y_pred = rf_model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# 特征重要性
importances = rf_model.feature_importances_
feature_names = features.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.sort_values(by='Importance', ascending=False))
plt.title('Feature Importance')
plt.show()
