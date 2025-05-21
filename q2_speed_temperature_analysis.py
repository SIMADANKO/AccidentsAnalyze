import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号无法显示的问题

# 读取数据
data = pd.read_csv('Sacramento_2022_Accidents.csv')  # 请确保文件路径正确

# 选择分析字段
df = data[['Speed', 'M_tem', 'L_tem']]

# 去除缺失值
df = data[['Speed', 'M_tem', 'L_tem']].copy()
df.dropna(inplace=True)

# 描述性统计（可选）
print("描述性统计：")
print(df.describe())

# 可视化相关性
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("变量间相关系数热力图")
plt.show()

# 散点图可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df = data[['Speed', 'M_tem', 'L_tem']].copy()
df.dropna(inplace=True)

plt.subplot(1, 2, 2)
sns.scatterplot(x='L_tem', y='Speed', data=df)
plt.title("Speed vs L_tem")
plt.tight_layout()
plt.show()

# 建立多元线性回归模型
X = df[['M_tem', 'L_tem']]
y = df['Speed']

model = LinearRegression()
model.fit(X, y)

# 输出回归系数
print("\n回归方程：")
print(f"Intercept（截距）: {model.intercept_:.4f}")
print(f"Coefficient of M_tem: {model.coef_[0]:.4f}")
print(f"Coefficient of L_tem: {model.coef_[1]:.4f}")

# 模型评估
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
print(f"\n模型评估：R² = {r2:.4f}, RMSE = {rmse:.4f}")

# 残差分析
residuals = y - y_pred
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True)
plt.title("残差分布图")
plt.xlabel("Residuals")
plt.show()

