import numpy as np
import pandas as pd
import cv2
from typing import Callable
import torch.utils.data
import matplotlib.pyplot as plt   

from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.linear_model import LinearRegression    # Basic Linear Regression
from sklearn.linear_model import Ridge               # Linear Regression with L2 regularization

from sklearn.model_selection import KFold            # Cross-validation tools

from sklearn.preprocessing import PolynomialFeatures # Feature transformations
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from io import StringIO

from pathlib import Path
import random
import os
import seaborn as sns

LEGEND_PATH = Path("facial_expressions/data/legend.csv")
IMG_DIR = IMG_DIR = Path("facial_expressions/images")
SEED = 1234

np.random.seed(SEED)
random.seed(SEED)

X = []
Y = []
CLASSES = sorted(['anger', 'fear', 'sadness', 'neutral', 'happiness', 'surprise', 'contempt', 'disgust'])

legend = pd.read_csv(LEGEND_PATH)
legend['emotion'] = legend['emotion'].str.lower()
legend = legend[legend['emotion'].isin(CLASSES)]
        # Reset indices
legend = legend.reset_index(drop = True)
        # Drop unecessary columns
legend = legend[['image', 'emotion']]



for img in os.listdir(IMG_DIR):
    img_array = cv2.imread(os.path.join(IMG_DIR,img), 0)
    img_resized=np.resize(img_array,(150,150,3))
    mask = legend[legend.iloc[:, 0] == img]
    if not mask.empty: #skip over images without a label
        X.append(img_resized.flatten())
        value = mask.iloc[0, 1]
        Y.append(CLASSES.index(value))

X = np.array(X)
Y = np.array(Y)

print(np.array(np.unique(Y, return_counts=True)).T)

print(X.shape)
print(Y.shape)
"""
X_tr, X_test, Y_tr, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1234, stratify=Y)


error_train = [] #between Y_tr and knn.predict(X_tr)
error_eval = [] #between Y_test and knn.predict(X_test)

for k in range(1, 11):
    knn_classifier = KNeighborsClassifier(n_neighbors=k).fit(X_tr, Y_tr)
    error_train.append(np.mean(knn_classifier.predict(X_tr) != Y_tr))
    error_eval.append(np.mean(knn_classifier.predict(X_test) != Y_test))

knn_classifier = KNeighborsClassifier(n_neighbors=9).fit(X_tr, Y_tr)
print(error_train)
print(error_eval)
print("--------------------------------\n")
confusion = confusion_matrix(Y_test, knn_classifier.predict(X_test))

sns.heatmap(confusion/np.sum(confusion), annot=True, 
            fmt='.2%', cmap='Blues')
plt.show()

k = len(error_train)

X_axis = np.arange(k)

plt.bar(X_axis - 0.2, error_train, 0.4, label = 'Girls') 
plt.bar(X_axis + 0.2, error_eval, 0.4, label = 'Boys') 
  
plt.xticks(X_axis, range(1, 11)) 

plt.title("Error Rate vs different K values (greyscale)")
plt.xlabel("K value")
plt.ylabel("Error Rate")
plt.legend(['Training Error', 'Evaluation Error'], loc='upper left')
plt.ylim(bottom = -0.1, top = 0.75)
plt.show()"""
