import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_excel("AirQualityUCI.xlsx", sheet_name="AirQualityUCI")

dataset['Date'] = pd.to_datetime(dataset['Date'], format='%d%m%Y')
dataset['Day'] = dataset['Date'].dt.day.astype(float)
dataset['Month'] = dataset['Date'].dt.month.astype(float)
dataset['Year'] = dataset['Date'].dt.year.astype(float)
dataset['Hour'] = dataset['Time'].apply(lambda x: x.hour).astype(float)

corr_matrix = dataset.corr(numeric_only=True)  

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.show()
