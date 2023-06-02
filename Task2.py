import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

data = pd.read_csv('dataset.csv')

X = data.drop('sl', axis=1)
y = data['sl']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = GaussianNB()

nb_classifier.fit(X_train, y_train)

print('Metrics on Test Set:')
print('-' * 20)
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
print('Accuracy:', accuracy)
print('Precision:', precision)

print('\nCorrelation Matrix:')
print('-' * 20)
corr_matrix = confusion_matrix(y_test, y_pred)
print(corr_matrix)
