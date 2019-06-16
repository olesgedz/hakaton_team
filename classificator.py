import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.externals import joblib

import pandas as pd

import sys

​

# For plots, not very necessary

# error = []

#

# # Calculating error for K values between 1 and 40

# for i in range(1, 40):

# knn = KNeighborsClassifier(n_neighbors=i)

# knn.fit(X_train, y_train)

# pred_i = knn.predict(X_test)

# error.append(np.mean(pred_i != y_test))

#

# plt.figure(figsize=(12, 6))

# plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',

# markerfacecolor='blue', markersize=10)

# plt.title('Error Rate K Value')

# plt.xlabel('K Value')

# plt.ylabel('Mean Error')

# plt.show()

​

​

def classificate(url, names, neighbors_num):

# Read dataset to pandas dataframe

dataset = pd.read_csv(url, names=names)

​

X = dataset.iloc[ :, :-3].values

y = dataset.iloc[ :, 4].values

​

# with pd.option_context('display.max_rows', None, 'display.max_columns', None): # more options can be specified also

# print(X)

# print("\n")

​

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

​

classifier = KNeighborsClassifier(n_neighbors=neighbors_num)

classifier.fit(x_train, y_train)

​

y_pred = classifier.predict(x_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

​

# Output a pickle file for the model

joblib.dump(classifier, 'saved_model.pkl')

print("---------Dump saved in saved_model.pkl---------\n")

​

​

# Load the pickle file

# clf_load = joblib.load('saved_model.pkl')

# assert classifier.score(X, y) == clf_load.score(X, y)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None): # more options can be specified also

# print(clf_load)

​

if __name__ == '__main__':

if (len(sys.argv) < 3):

print("usage: python3 classificator.py 'path_or_url_to_csv' number_of_neighbors")

exit(0)

url = sys.argv[1]#"/Users/aelinor-/Downloads/iris.data"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

​

classificate(url, names, int(sys.argv[2]))

