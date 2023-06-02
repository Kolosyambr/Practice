import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Загрузка данных
data = pd.read_csv('dataset.csv')

# Разделение на признаки (X) и целевую переменную (y)
X = data.drop('sl', axis=1)
y = data['sl']

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание объекта классификатора AdaBoostClassifier
adaboost_classifier = AdaBoostClassifier()

# Обучение модели на тренировочных данных
adaboost_classifier.fit(X_train, y_train)

# Предсказание классов на тестовых данных
y_pred = adaboost_classifier.predict(X_test)

# Вычисление accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Вычисление precision
precision = precision_score(y_test, y_pred, average='macro')
print('Precision:', precision)

# Вычисление корреляционной матрицы
corr_matrix = confusion_matrix(y_test, y_pred)
print('Correlation Matrix:')
print(corr_matrix)

# Задание сетки параметров для перебора
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.5, 1.0],
    'algorithm': ['SAMME', 'SAMME.R']
}

# Создание объекта GridSearchCV
grid_search = GridSearchCV(adaboost_classifier, param_grid, cv=5)

# Подгонка модели на тренировочных данных
grid_search.fit(X_train, y_train)

# Вывод наилучших параметров модели
print('Best Parameters:', grid_search.best_params_)
