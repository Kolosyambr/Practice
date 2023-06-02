import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# Загрузка данных
data = pd.read_csv('dataset.csv')

# Разделение на признаки (X) и целевую переменную (y)
X = data.drop('sl', axis=1)
y = data['sl']

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание объекта классификатора KNN
knn_classifier = KNeighborsClassifier()

# Задание сетки параметров для перебора
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

# Создание объекта GridSearchCV
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)

# Подгонка модели на тренировочных данных
grid_search.fit(X_train, y_train)

# Вывод наилучших параметров модели
print('Best Parameters:', grid_search.best_params_)

# Обучение модели на тренировочных данных с оптимальными параметрами
knn_classifier = KNeighborsClassifier(**grid_search.best_params_)
knn_classifier.fit(X_train, y_train)

# Предсказание классов на тестовых данных
y_pred = knn_classifier.predict(X_test)

# Вычисление метрик accuracy и precision
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
print('Accuracy:', accuracy)
print('Precision:', precision)

# Вычисление корреляционной матрицы
corr_matrix = confusion_matrix(y_test, y_pred)
print('Correlation Matrix:')
print(corr_matrix)
