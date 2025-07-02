import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 读取数据
df = pd.read_excel(r"F:\数据和代码下载\数据论文\测算-统计数据对比.xlsx")
#——————————————————————————————————————Internal consistency validation


months = [str(i) for i in range(1, 13)]
month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
import string

sns.set(style="whitegrid", font_scale=1.2)
# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
# Nature 风格颜色
scatter_color = "#0072B2"
line_color = "#D55E00"
ci_alpha = 0.25
reference_line_color = "#666666"

fig, axs = plt.subplots(3, 4, figsize=(20, 15))
axs = axs.flatten()

labels = [chr(ord('a') + i) for i in range(12)]  # ['a', 'b', ..., 'l']

for i in range(1, 13):
    M_col = f'M{i}'
    TM_col = f'TM{i}'

    x_raw = df[TM_col].values
    y = df[M_col].values
    x = x_raw.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    r2 = r2_score(y, y_pred)
    bias = np.mean(y - x_raw)
    t_stat, p_val = stats.ttest_rel(y, x_raw)

    x_fit = np.linspace(min(x_raw), max(x_raw), 100).reshape(-1, 1)
    y_fit = model.predict(x_fit)

    n = len(x_raw)
    y_err = y - y_pred
    se = np.sqrt(np.sum(y_err ** 2) / (n - 2))
    x_mean = np.mean(x_raw)
    t_val = stats.t.ppf(0.975, df=n - 2)

    conf_interval = t_val * se * np.sqrt(1/n + (x_fit.flatten() - x_mean) ** 2 / np.sum((x_raw - x_mean) ** 2))
    y_upper = y_fit + conf_interval
    y_lower = y_fit - conf_interval

    ax = axs[i - 1]
    ax.grid(False)

    ax.scatter(x, y, color=scatter_color, alpha=0.6, label='Data points')
    ax.plot(x_fit, y_fit, color=line_color, label='Linear fit', linewidth=2)
    ax.fill_between(x_fit.flatten(), y_lower, y_upper, color=line_color, alpha=ci_alpha, label='95% CI')
    ax.plot(np.sort(x_raw), np.sort(x_raw), linestyle='--', color=reference_line_color, label='1:1 line')

    ax.set_title(month_names[i - 1], fontsize=14, fontname='Times New Roman')
    ax.set_xlabel('Statistical Value (100k MWh)', fontsize=14, fontname='Times New Roman')
    ax.set_ylabel('Estimated Value (100k MWh)', fontsize=14, fontname='Times New Roman')

    ax.text(0.05 * max(x_raw), 0.85 * max(y),
            f"$R^2$ = {r2:.3f}\nBias = {bias:.2f}\np = {p_val:.1e}",
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
            fontname='Times New Roman')

    ax.legend()

    ax.text(-0.1, 1.05, labels[i - 1], transform=ax.transAxes,
            fontsize=16, fontweight='bold', fontname='Times New Roman',
            va='top', ha='right')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])


plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, linregress


real = np.array([7232, 6235, 6944, 6362, 6716, 7451, 8324, 8520, 7092, 6834, 7630, 7032])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
estimated = np.array([df[f"M{i}"].sum() for i in range(1, 13)])

mape = np.abs((estimated - real) / real) * 100
ae = np.abs(estimated - real)

monthly_p, monthly_t = [], []
for i in range(1, 13):
    est_city = df[f"M{i}"]
    ratio = real[i - 1] / estimated[i - 1]
    true_city = est_city * ratio
    t_stat, p_val = ttest_rel(est_city, true_city)
    monthly_t.append(t_stat)
    monthly_p.append(p_val)


sns.set(style="white", font_scale=1.5)
palette = ['#B7B7EB', '#F09BA0', '#EAB883']
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['text.color'] = 'black'

fig, ax = plt.subplots(figsize=(8, 7))
x = np.arange(12)
bar_width = 0.35
ax.bar(x - bar_width / 2, real, width=bar_width, label='Statistical', color=palette[0])
ax.bar(x + bar_width / 2, estimated, width=bar_width, label='Estimated', color=palette[1])
ax.set_xticks(x)
ax.set_xticklabels(months) #,rotation=25
ax.set_ylabel('Electricity (100k MWh)')
legend = ax.legend(loc='upper left', frameon=True)
legend.get_frame().set_edgecolor('gray')
legend.get_frame().set_linewidth(0.8)

for spine in ax.spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(0.8)

ax.grid(False)
fig.tight_layout()
plt.close()

fig, ax = plt.subplots(figsize=(8, 7))
sns.lineplot(x=months, y=mape, marker='o', color=palette[2], linewidth=2, ax=ax)
for i, val in enumerate(mape):
    ax.text(i, val + 0.8, f'{val:.1f}%', ha='center', fontsize=12)
ax.set_ylabel('MAPE (%)')
ax.set_ylim(0, max(mape) + 8)

# 设置灰色边框
for spine in ax.spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(0.8)

ax.grid(False)
fig.tight_layout()
plt.close()


df['Estimated_Annual'] = df[[f"M{i}" for i in range(1, 13)]].sum(axis=1)
df['Actual_Annual'] = df['year']
df['Abs_Error'] = np.abs(df['Estimated_Annual'] - df['Actual_Annual'])
df['MAPE'] = (df['Abs_Error'] / df['Actual_Annual']) * 100

fig, ax = plt.subplots(figsize=(8, 7))
sns.regplot(x='Actual_Annual', y='Estimated_Annual', data=df,
            scatter_kws={'s': 40, 'alpha': 0.7},
            line_kws={'color': 'black'},
            ax=ax)

slope, intercept, r_value, p_value, std_err = linregress(df['Actual_Annual'], df['Estimated_Annual'])
ax.text(0.05, 0.92, f'$R^2$ = {r_value**2:.3f}', transform=ax.transAxes, fontsize=12)
ax.text(0.05, 0.86, f'p = {p_value:.4f}', transform=ax.transAxes, fontsize=12)
ax.set_xlabel('Actual Annual (100k MWh)')
ax.set_ylabel('Estimated Annual (100k MWh)')

for spine in ax.spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(0.8)

ax.grid(False)
fig.tight_layout()
plt.close()

#-------------------------Model performance evaluation-------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, f_oneway


np.random.seed(42)
models = ['Decision Tree', 'Linear', 'K-nearest neighbors', 'Random Forest']
folds = np.tile(np.arange(1, 11), len(models))

cv_results = {
    'Model': np.repeat(models, 10),
    'Fold': folds,
    'MAE': np.concatenate([
        [0.2038, 0.2193, 0.1832, 0.1831, 0.1995, 0.1789, 0.1774, 0.2128, 0.1553, 0.1688],
        [0.465, 0.4154, 0.3665, 0.4119, 0.4091, 0.4064, 0.3655, 0.3925, 0.3831, 0.4054],
        [0.4126, 0.3753, 0.3367, 0.3876, 0.3756, 0.3649, 0.3782, 0.3547, 0.3831, 0.3671],
        [0.1209, 0.1241, 0.1167, 0.1219, 0.1165, 0.1108, 0.1076, 0.1323, 0.0976, 0.1135]
    ]),
    'RMSE': np.concatenate([
        [0.449, 0.5598, 0.4399, 0.3755, 0.5018, 0.2941, 0.286, 0.6266, 0.2703, 0.3477],
        [0.7873, 0.7257, 0.6083, 0.7211, 0.744, 0.5712, 0.5138, 0.7601, 0.5605, 0.6045],
        [0.7749, 0.724, 0.6054, 0.7342, 0.7269, 0.5709, 0.5528, 0.7639, 0.6094, 0.6152],
        [0.2035, 0.3041, 0.2165, 0.2165, 0.285, 0.1808, 0.1822, 0.4649, 0.1704, 0.2088]
    ]),
    'R2': np.concatenate([
        [0.7315, 0.5001, 0.6015, 0.787, 0.6197, 0.7911, 0.8029, 0.4296, 0.8239, 0.7557],
        [0.1746, 0.1601, 0.238, 0.2145, 0.164, 0.2121, 0.364, 0.1604, 0.2424, 0.2614],
        [0.2003, 0.164, 0.2455, 0.1857, 0.2019, 0.2129, 0.2636, 0.1521, 0.1043, 0.2348],
        [0.9448, 0.8525, 0.9035, 0.9254, 0.8773, 0.9211, 0.92, 0.6859, 0.93, 0.9119]
    ])
}
df = pd.DataFrame(cv_results)

sns.set(style="white", font_scale=1.2)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17
plt.rcParams['legend.fontsize'] = 17
plt.rcParams['text.color'] = 'black'
# 配色与样式
colors = {
    'Decision Tree': '#B7B7EB',
    'Linear': '#F09BA0',
    'K-nearest neighbors': '#EAB883',
    'Random Forest': '#7DCFB6'
}
markers = {
    'Decision Tree': '*',
    'Linear': 'X',
    'K-nearest neighbors': 'D',
    'Random Forest': 'o'
}
palette = [colors[m] for m in models]

metric_labels = {
    'MAE': 'MAE (100k MWh)',
    'RMSE': 'RMSE (100k MWh)',
    'R2': 'R²'
}



def pval_to_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

for i, metric in enumerate(['MAE', 'RMSE', 'R2']):
    plt.figure(figsize=(8, 6))

    # 箱线图，设置透明度
    ax = sns.boxplot(
        x='Model', y=metric, data=df, hue='Model',
        palette=palette, width=0.6, fliersize=0,
        linewidth=1.2, boxprops=dict(alpha=0.3), legend=False
    )

    # 叠加散点
    for model in models:
        subset = df[df['Model'] == model]
        x_jitter = np.random.normal(loc=models.index(model), scale=0.06, size=len(subset))
        plt.scatter(
            x_jitter,
            subset[metric],
            color=colors[model],
            edgecolor='black',
            s=80,
            marker=markers[model],
            alpha=0.85
        )


    plt.ylabel(metric_labels[metric])
    plt.xlabel('')
    plt.grid(False)


    for spine in plt.gca().spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.show()

#-------------------------Model performance evaluation-------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import DBSCAN


# df.set_index('城市', inplace=True)

cities = df['城市']
X = df.drop(columns=['城市'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertia = []
k_range = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia (Within-cluster SSE)')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()


kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)


df['Cluster'] = labels







import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_map = {f'M{i}': month_labels[i - 1] for i in range(1, 13)}
df_renamed = df.rename(columns=month_map)


df_melted = df_renamed.melt(id_vars=['市', 'Cluster'],
                            value_vars=month_labels,
                            var_name='Month',
                            value_name='Electricity')


palette = {
    0: "#FF9999",
    1: "#FFE699",
    2: "#D4FFCC",
    3: "#9CE6FF",
}


sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['text.color'] = 'black'


plt.figure(figsize=(14, 10))
ax = sns.boxplot(
    data=df_melted,
    x='Month',
    y='Electricity',
    hue='Cluster',
    palette=palette,
    hue_order=[0, 1, 2, 3],
    width=0.6,
    boxprops=dict(edgecolor='#E1E1E1'),
    showfliers=False
)


quarter_colors = ['#e6f7ff', '#fff7e6', '#e6ffe6', '#f9e6ff']
quarter_labels = ['Spring', 'Summer', 'Fall', 'Winter']
for i in range(4):
    start = i * 3 - 0.5
    end = start + 3
    ax.axvspan(start, end, facecolor=quarter_colors[i], alpha=0.3, zorder=0)


handles, labels = ax.get_legend_handles_labels()
new_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
ax.legend(handles=handles, labels=new_labels, title=None, loc='upper right',bbox_to_anchor=(1, 0.9))

# 图形美化
plt.xlabel("")
plt.ylabel("Electricity Consumption (100k MWh)")
plt.grid(False)
ax.set_xlim(-0.4, 11.4)
# 调整y轴最大值
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax * 1.5)

plt.tight_layout()
plt.show()


