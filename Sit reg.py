import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# 定义函数：读取和预处理数据
def load_and_preprocess_data(file_path):
    # 读取数据
    data = pd.read_csv(file_path).dropna()

    # 定义天气特征列名
    weather_features = [
        'Temperature Max', 'Temperature Min', 'Temperature', 'Dew Point', 'Humidity',
        'Precipitation', 'Precipitation Cover', 'Wind Gust', 'Wind Speed', 'Pressure',
        'Cloud Cover', 'Solar Radiation', 'Solar Energy'
    ]

    # 生成滞后特征
    for feature in weather_features:
        data[f'{feature}_lag1'] = data[feature].shift(1)
        data[f'{feature}_lag2'] = data[feature].shift(2)

    # 删除前两天的数据，因为它们没有滞后特征
    data = data.dropna()

    return data

# 指定文件路径
file_path = 'C:/Users/shang/Desktop/IC/Sem3/All/Situse.csv'  # 请根据需要修改文件路径

# 读取并处理数据
data = load_and_preprocess_data(file_path)

# 特征工程，删除不需要的列
features = data.drop(columns=['ZeroTimes', 'City', 'Date'])
target = data['ZeroTimes']

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.2, random_state=42)

# 建立四层神经网络模型
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1))  # 输出层

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# 训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32,
                    callbacks=[early_stopping, reduce_lr])

# 评估模型
y_pred = model.predict(X_test).flatten()  # 将预测值转换为一维数组
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mean_actual = np.mean(y_test)
# 计算相对均方误差
relative_mse = mse / np.mean(y_test)

# 输出评估结果
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")
print(f"Relative Mean Squared Error: {relative_mse}")
print(f"Mean Actual: {mean_actual}")

# 可视化训练过程
plt.figure(figsize=(14, 6))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot actual vs predicted with fit line
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0, max(y_test)], [0, max(y_test)], 'k--', lw=2)  # 这条线是 y=x 的参考线

# 计算拟合线
z = np.polyfit(y_test, y_pred, 1)  # 1 表示线性拟合
p = np.poly1d(z)

# 绘制拟合线
plt.plot(y_test, p(y_test), 'r-', lw=2)

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Test Data: Actual vs Predicted')
plt.legend(['Data', 'y=x', 'Fit line'], loc='upper left')

plt.show()

