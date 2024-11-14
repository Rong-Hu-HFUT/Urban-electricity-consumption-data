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

