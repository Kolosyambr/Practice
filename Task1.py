import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('dataset.csv')

columns_to_normalize = ['sr', 'rr', 't', 'lm', 'bo', 'rem', 'sh', 'hr']

scaler = MinMaxScaler()

data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

sns.countplot(x='sl', data=data)
plt.title('Distribution of Stress Levels')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='sr', data=data)
plt.title('Distribution of Snoring Rate')
plt.show()

data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

data[columns_to_normalize].hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

corr_matrix = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()