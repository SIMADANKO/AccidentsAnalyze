from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
import matplotlib.pyplot as plt

# 创建 SparkSession
spark = SparkSession.builder \
    .appName("Q1 Accident Analysis") \
    .master("local[*]") \
    .getOrCreate()

# 读取 CSV 文件（只选取需要的列）
df = spark.read.csv("Sacramento_2022_Accidents.csv", header=True, inferSchema=True) \
    .select("Accident", "Day", "Week", "Weather")

# 只保留发生事故的数据（Accident == 1）
accidents = df.filter(col("Accident") == 1)

# 哪一天发生事故最多（按日期）
print("哪一天发生事故最多：")
day_counts = accidents.groupBy("Day").agg(count("*").alias("accident_count")).orderBy(col("accident_count").desc())
day_counts.show(1)

# 将结果转为 Pandas 并画图
day_pdf = day_counts.toPandas()
plt.figure(figsize=(10,5))
plt.bar(day_pdf['Day'], day_pdf['accident_count'], color='skyblue')
plt.xlabel("Day of the Month")
plt.ylabel("Number of Accidents")
plt.title("Accidents by Day of the Month")
plt.xticks(day_pdf['Day'])
plt.tight_layout()
plt.show()

# 星期几发生事故最多
print("星期几发生事故最多：")
week_counts = accidents.groupBy("Week").agg(count("*").alias("accident_count")).orderBy(col("accident_count").desc())
week_counts.show(1)

# 转换并画图
week_pdf = week_counts.toPandas()
plt.figure(figsize=(8,5))
plt.bar(week_pdf['Week'], week_pdf['accident_count'], color='orange')
plt.xlabel("Day of the Week")
plt.ylabel("Number of Accidents")
plt.title("Accidents by Day of the Week")
plt.tight_layout()
plt.show()

# 哪种天气状况下事故最多
print("哪种天气状况下事故最多：")
weather_counts = accidents.groupBy("Weather").agg(count("*").alias("accident_count")).orderBy(col("accident_count").desc())
weather_counts.show(1)

# 转换并画图
weather_pdf = weather_counts.toPandas()
plt.figure(figsize=(10,5))
plt.bar(weather_pdf['Weather'], weather_pdf['accident_count'], color='green')
plt.xlabel("Weather Condition")
plt.ylabel("Number of Accidents")
plt.title("Accidents by Weather Condition")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 停止 SparkSession
spark.stop()
