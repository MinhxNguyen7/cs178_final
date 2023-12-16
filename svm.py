import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt   

from sklearn.model_selection import train_test_split

from pathlib import Path
import random
import os

from sklearn import svm 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

LEGEND_PATH = Path("facial_expressions/data/legend.csv")
IMG_DIR = IMG_DIR = Path("facial_expressions/images")
SEED = 1234

np.random.seed(SEED)
random.seed(SEED)

X = []
Y = []
CLASSES = sorted(['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'])

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

print(X.shape)
print(Y.shape)

X_tr, X_test, Y_tr, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1234, stratify=Y)


# Defining the parameters grid for GridSearchCV 
param_grid={'C':[0.1,1,10,100], 
            'gamma':[0.0001,0.001,0.1,1], 
            'kernel':['rbf','poly']} 
  
# Creating a support vector classifier 
svc=svm.SVC(probability=True) 
  
# Creating a model using GridSearchCV with the parameters grid 
model=GridSearchCV(svc,param_grid).fit(X_tr, Y_tr)

print(model.best_params_) 
# Testing the model using the testing data 
Y_pred = model.predict(X_test) 
  
# Calculating the accuracy of the model 
accuracy = accuracy_score(Y_pred, Y_test) 
  
# Print the accuracy of the model 
print(f"The model is {accuracy*100}% accurate")