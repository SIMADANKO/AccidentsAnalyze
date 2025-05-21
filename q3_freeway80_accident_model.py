import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier  # ✅ 使用 XGBoost

# 读取数据
df = pd.read_csv('Sacramento_2022_Accidents.csv')

# 筛选80号高速公路数据
df_80 = df[df['Freeway'] == 80]

# 删除缺失值
df_80 = df_80.dropna(subset=['Speed', 'M_tem', 'L_tem', 'Day', 'Week', 'Windscale',
                             'Latitude', 'Longitude', 'Weather', 'Direction',
                             'ID', 'Freeway', 'Timestamp', 'Accident'])

# 编码方向字段
df_80['Direction'] = df_80['Direction'].map({'N': 0, 'S': 1, 'E': 2, 'W': 3})

# 处理时间戳
df_80['Timestamp'] = pd.to_datetime(df_80['Timestamp'])
df_80['Hour'] = df_80['Timestamp'].dt.hour
df_80['Minute'] = df_80['Timestamp'].dt.minute

# 特征与标签
features = ['Speed', 'M_tem', 'L_tem', 'Day', 'Week', 'Windscale',
            'Latitude', 'Longitude', 'Weather', 'Direction', 'ID', 'Freeway', 'Hour', 'Minute']
X = df_80[features]
y = df_80['Accident']

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE 处理
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 训练/测试集划分
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# ✅ 使用 XGBoost 训练模型
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

# 预测
y_pred = xgb.predict(X_test)

# 评估结果
print("✅ 准确率：", accuracy_score(y_test, y_pred))
print("\n📊 分类报告：\n", classification_report(y_test, y_pred))
print("🔢 混淆矩阵：\n", confusion_matrix(y_test, y_pred))

# 特征重要性图
importances = xgb.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [features[i] for i in indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names, palette='viridis')
plt.title("📌 FeaturesImportance - XGBoost")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# 混淆矩阵热力图
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()

