import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# Загрузка данных
data = pd.read_csv('dataset.csv')

# Разделение на признаки (X) и целевую переменную (y)
X = data.drop('sl', axis=1)
y = data['sl']

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели дерева решений
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Предсказание классов на тестовых данных
y_pred = dt_classifier.predict(X_test)

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

# Подбор оптимальных параметров модели
param_grid = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

grid_search = GridSearchCV(dt_classifier, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print('Best Parameters:', grid_search.best_params_)
