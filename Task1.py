import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Загрузка данных
data = pd.read_csv('dataset.csv')

# Выбор столбцов для нормализации
columns_to_normalize = ['sr', 'rr', 't', 'lm', 'bo', 'rem', 'sh', 'hr']

# Создание объекта MinMaxScaler
scaler = MinMaxScaler()

# Применение нормализации
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# Распределение целевого класса
sns.countplot(x='sl', data=data)
plt.title('Distribution of Stress Levels')
plt.show()

# Анализ категориального признака "snoring rate"
plt.figure(figsize=(8, 6))
sns.countplot(x='sr', data=data)
plt.title('Distribution of Snoring Rate')
plt.show()

# Применение нормализации
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# Визуализация нормализованных данных в виде гистограммы
data[columns_to_normalize].hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Вычисление матрицы корреляции
corr_matrix = data.corr()

# Визуализация корреляции в виде тепловой карты
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()