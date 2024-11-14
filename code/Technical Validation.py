
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20


df = pd.read_csv(r"F:\数据和代码下载\数据论文\202408extrapolation_results - 春.csv", encoding='ISO-8859-1')
columns_of_interest = ['Temp_min', 'Temp_Ave', 'Wind', 'NTL', 'Electricity']


# 绘制相关性热力图
plt.subplots(figsize=(10, 10))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(df[columns_of_interest].corr(), annot=True, vmax=1, square=True, cmap="Blues")
plt.savefig(r'F:\数据和代码下载\数据论文\相关性 - 春.png',dpi=300)


df = pd.read_csv(r"F:\数据和代码下载\数据论文\202408extrapolation_results - 夏.csv", encoding='ISO-8859-1')
columns_of_interest = ['Temp_min', 'Temp_Ave', 'Wind', 'NTL', 'Electricity']


plt.subplots(figsize=(10, 10))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(df[columns_of_interest].corr(), annot=True, vmax=1, square=True, cmap="Blues")
plt.savefig(r'F:\数据和代码下载\数据论文\相关性 - 夏.png',dpi=300)

df = pd.read_csv(r"F:\数据和代码下载\数据论文\202408extrapolation_results - 秋.csv", encoding='ISO-8859-1')
columns_of_interest = ['Temp_min', 'Temp_Ave', 'Wind', 'NTL', 'Electricity']

# 绘制相关性热力图
plt.subplots(figsize=(10, 10))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(df[columns_of_interest].corr(), annot=True, vmax=1, square=True, cmap="Blues")
# plt.title('相关性热力图')
plt.savefig(r'F:\数据和代码下载\数据论文\相关性 - 秋.png',dpi=300)

df = pd.read_csv(r"F:\数据和代码下载\数据论文\202408extrapolation_results - 冬.csv", encoding='ISO-8859-1')
columns_of_interest = ['Temp_min', 'Temp_Ave', 'Wind', 'NTL', 'Electricity']


plt.subplots(figsize=(10, 10))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(df[columns_of_interest].corr(), annot=True, vmax=1, square=True, cmap="Blues")
plt.savefig(r'F:\数据和代码下载\数据论文\相关性 - 冬.png',dpi=300)

plt.show()

