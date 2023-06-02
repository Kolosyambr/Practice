import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('dataset.csv')

X = data.drop('sl', axis=1)
y = data['sl']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

qda_classifier = QuadraticDiscriminantAnalysis()

qda_classifier.fit(X_train, y_train)

y_pred = qda_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

precision = precision_score(y_test, y_pred, average='macro')
print('Precision:', precision)

corr_matrix = confusion_matrix(y_test, y_pred)
print('Correlation Matrix:')
print(corr_matrix)

param_grid = {
    'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
}

grid_search = GridSearchCV(qda_classifier, param_grid, cv=5)

grid_search.fit(X_train, y_train)

print('Best Parameters:', grid_search.best_params_)
