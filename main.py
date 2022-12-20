import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('teleCust1000t.csv')
print('-----df.head-----')
print(df.head())
print('----custcat value counts-----')
print(df['custcat'].value_counts())
print('-----df.columns-----')
print(df.columns)
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values
print('-----X------')
print(X[0:5])
print('-----custcat values------')
y = df['custcat'].values
print(y[0:5])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

decisiontree = DecisionTreeClassifier(random_state=0)
model = decisiontree.fit(X_train, y_train)

target_predicted = model.predict(X_test)
# print("Accuracy", model.score(X_test, y_test))

matrix = confusion_matrix(y_test, target_predicted)
# print("Class Confusion Matrix\n", matrix)

decisiontree_entropy = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=4, max_leaf_nodes=15)

model_entropy = decisiontree_entropy.fit(X_train, y_train)

target_predicted2 = model_entropy.predict(X_test)
print("Accuracy", model_entropy.score(X_test, y_test))

# 1  #4  #2  #2
prediction_test = [[2, 5, 33, 0, 10, 125.000, 4, 5, 0.000, 1, 1], [2, 57, 54, 1, 30, 115.000, 4, 23, 0.000, 1, 3],
                   [1, 40, 42, 1, 16, 108.000, 3, 17, 0.000, 0, 4], [3, 70, 50, 1, 29, 67.000, 1, 22, 0.000, 0, 4]]
prediction_result_test = [[1], [4], [2], [2]]
predicted3 = model_entropy.predict(prediction_test)
score3 = model_entropy.score(prediction_test, prediction_result_test)
print('prediction for custom array ', predicted3)
print('score for custom array',score3)

matrix2 = confusion_matrix(y_test, target_predicted2)
print("Class Confusion Matrix\n", matrix2)

import pydotplus
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image, display
from sklearn import tree

data_feature_names = ['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender',
                      'reside']

dot_data = tree.export_graphviz(model_entropy, out_file="resume.dot",
                                feature_names=data_feature_names, class_names=['1', '2', '3', '4'],
                                filled=True, rounded=True, special_characters=True, leaves_parallel=False)

graph = pydotplus.graphviz.graph_from_dot_file("resume.dot")
graph.write_png('tree.png')

Image(filename='tree.png')
