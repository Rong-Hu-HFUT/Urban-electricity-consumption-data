import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def max_min_scale(X, mins=None, maxs=None):
    return (X - mins) / (maxs - mins)


total_inference_df = pd.read_excel(r"E:\Dataset.xlsx")

total_inference_X = total_inference_df.drop(columns=['城', 'flag', 'Ele']) #


mins = total_inference_X.min(axis=0)
maxs = total_inference_X.max(axis=0)
total_inference_X = max_min_scale(total_inference_X, mins, maxs)


X_train = total_inference_X[total_inference_df["flag"] == 1]
y_train = total_inference_df[total_inference_df["flag"] == 1]["Ele"].values
X_predict = total_inference_X[total_inference_df["flag"] == 0]


rfr = RandomForestRegressor(n_estimators=100, min_samples_split=2, n_jobs=-1, random_state=1212)


kf = KFold(n_splits=10, shuffle=True, random_state=1212)


r2_scores = []
mae_scores = []
mse_scores = []

for train_index, test_index in kf.split(X_train):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

    rfr.fit(X_train_fold, y_train_fold)

    y_pred_fold = rfr.predict(X_test_fold)

    r2_scores.append(r2_score(y_test_fold, y_pred_fold))
    mae_scores.append(mean_absolute_error(y_test_fold, y_pred_fold))
    mse_scores.append(mean_squared_error(y_test_fold, y_pred_fold))


rfr.fit(X_train, y_train)
total_inference_df.loc[total_inference_df["flag"] == 0, "inference"] = rfr.predict(X_predict)

print("Cross-Validation R^2 Scores: ", r2_scores)
print("Mean CV R^2 Score: ", np.mean(r2_scores))
print("Cross-Validation MAE Scores: ", mae_scores)
print("Mean CV MAE Score: ", np.mean(mae_scores))
print("Cross-Validation MSE Scores: ", mse_scores)
print("Mean CV MSE Score: ", np.mean(mse_scores))
print("Total Inference Sum: ", total_inference_df["inference"].sum())


import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

def set_chinese_font():
    try:
        font_path = r"C:\Windows\Fonts\simhei.ttf"
        assert os.path.exists(font_path)
    except Exception:
        try:
            font_path = r"C:\Windows\Fonts\msyh.ttc"  # Microsoft YaHei
            assert os.path.exists(font_path)
        except Exception:
            font_path = None

    if font_path:
        font = FontProperties(fname=font_path, size=12)
        matplotlib.rcParams['font.family'] = font.get_name()
        matplotlib.rcParams['font.sans-serif'] = [font.get_name()]
        matplotlib.rcParams['axes.unicode_minus'] = False
    else:
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

set_chinese_font()


df = pd.read_excel("F:\数据和代码下载\数据论文\data.xlsx") 

features = ['平均最低气温', '平均最高气温', '平均气温', '平均风速', 'NTL-D-Ave']
target = 'Ele'


train_df = df[df['flag'] == 1].copy()
train_df = train_df[~train_df[target].isna()].reset_index(drop=True)

X = train_df[features].copy()
y = train_df[target].copy()

X = X.fillna(X.mean())

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=100,          # Number of estimators
    criterion='squared_error', # Criterion for split quality
    min_samples_split=2,       # Minimum number of samples required to split an internal node
    min_samples_leaf=1,        # Minimum samples at a leaf node
    max_features='sqrt',       # Max features for best split
    bootstrap=True,            # Bootstrap sampling
    oob_score=False,           # Out-of-bag score usage
    max_samples=None,          # Number of samples to draw for each tree
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_valid = rf.predict(X_valid)
baseline_mse = mean_squared_error(y_valid, y_pred_valid)
baseline_r2 = r2_score(y_valid, y_pred_valid)

print("验证集 R² Score:", baseline_r2)
print("验证集 MSE:", baseline_mse)


perm = permutation_importance(
    rf, X_valid, y_valid, n_repeats=40, random_state=42,
    scoring='neg_mean_squared_error', n_jobs=-1
)

imp_abs = perm.importances_mean  # 这是 increase_in_MSE（正值表示打乱导致 MSE 增大）
imp_std = perm.importances_std
feature_names = np.array(X_valid.columns)


if baseline_mse > 0:
    imp_pct = 100.0 * imp_abs / baseline_mse
else:
    imp_pct = imp_abs  

imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Imp_Increase_MSE': imp_abs,
    'Imp_STD': imp_std,
    'Imp_Increase_MSE_pct': imp_pct
}).sort_values('Imp_Increase_MSE_pct', ascending=False).reset_index(drop=True)

print("\nPermutation Importance (increase in MSE and % relative to baseline MSE):")
print(imp_df)
