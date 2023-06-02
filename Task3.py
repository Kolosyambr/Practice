import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

data = pd.read_csv('dataset.csv')

X = data.drop('sl', axis=1)
y = data['sl']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_classifier = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)

grid_search.fit(X_train, y_train)

print('Best Parameters:', grid_search.best_params_)

knn_classifier = KNeighborsClassifier(**grid_search.best_params_)
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
print('Accuracy:', accuracy)
print('Precision:', precision)

corr_matrix = confusion_matrix(y_test, y_pred)
print('Correlation Matrix:')
print(corr_matrix)
