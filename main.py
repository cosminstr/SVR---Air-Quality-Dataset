from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
  
air_quality = fetch_ucirepo(id=360) 
dataset = pd.DataFrame(air_quality.data.features)
# print(dataset.shape)
  
# CO = dataset['AH'].to_numpy()
# histCO, bins = np.histogram(CO, bins=60)
# plt.figure()
# plt.stairs(histCO, bins)
# plt.show()

# print(dataset) 
# print(target)
finalDataset = dataset.drop(columns=['Date', 'Time', 'CO(GT)', 'NMHC(GT)', 
                                     'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S5(O3)'])

finalDataset = finalDataset[~((finalDataset['PT08.S1(CO)'] <= 350) | (finalDataset['PT08.S2(NMHC)'] <= 350)
                              | (finalDataset['PT08.S4(NO2)'] <= 350) | (finalDataset['T'] <= -5) | 
                              (finalDataset['RH'] <= -5) | (finalDataset['AH'] <= -5))]
target = finalDataset[['C6H6(GT)']]
finalDataset = finalDataset.drop(columns=['C6H6(GT)'])

print(finalDataset)
print(target)

X = finalDataset.to_numpy() 
y = target.to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr = SVR(kernel='rbf', C=128, gamma = 0.0078125, epsilon=0.001953125)  
# svr = SVR(kernel='poly', degree=3, C=128, gamma = 'auto') #0.0078125, epsilon=0.001953125)  
svr.fit(X_train_scaled, y_train)
y_pred = svr.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")

print("-------------------------------------------")

y_train_pred = svr.predict(X_train_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"train MSE: {train_mse}")
print(f"train MAE: {train_mae}")
print(f"train r2: {train_r2}")

print(f"test MSE: {mse}")
print(f"test MAE: {mae}")
print(f"test r2: {r2}")


