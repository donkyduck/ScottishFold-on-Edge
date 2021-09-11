# Machine Learning model 3 (Test w/ some different MAC address + K.Arunan's Echo)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from catboost import CatBoostClassifier,Pool
import os
import time
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV

TRAIN_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model3_dataset", "model3_train_paper.csv")
TEST_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_dns", "experiment_dns.csv")
MODEL_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_dns", "experiment_dns.cbm")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]),   range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Train dataset
headernames = ['ipAddr', 'mac', 'portSrc', 'portDst', 'pktLength', 'deviceName', 'protocol', 'detail' ]
dataset = pd.read_csv(TRAIN_DATASET, names = headernames)
dataset.head()
X_train = dataset.drop(['deviceName'],axis=1)
X_train = X_train.drop(['detail'],axis=1)
y_train = dataset['deviceName']
#########################################################################
# Test dataset
testDataset = pd.read_csv(TEST_DATASET, names = headernames)
testDataset.head()
X_test = testDataset.drop(['deviceName'],axis=1)
X_test = X_test.drop(['detail'],axis=1)
y_test = testDataset['deviceName']

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
# IMPORTANT! : cannot use two CatBoost pool.
pool = Pool(X_train, y_train,cat_features=categorical_features_indices)

params = {
    'iterations': 100,
    'learning_rate': 0.59,
    'eval_metric': 'Accuracy',
    'l2_leaf_reg':3,
    'use_best_model': False
    }

classifier = CatBoostClassifier(**params)
start = time.time()
classifier.fit(pool)
stop = time.time()

start2 = time.time()
y_pred = classifier.predict(X_test)
stop2 = time.time()

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
report = classification_report(y_test, y_pred)
print("Classification Report:",)
print (report)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
traningTime = str(stop - start)
print("Training time: " + traningTime)
testingTime = str(stop2 - start2)
print("Testing time: " + testingTime)

feature_importances = classifier.get_feature_importance(pool)
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))

className = ["Device 0","Device 1","Device 2","Device 3","Device 4","Device 5","Device 6","Device 7"]
plt.figure()
plot_confusion_matrix(cm, classes=className,normalize=True,
                      title="DNS packets dataset's confusion matrix")
plt.show()

classifier.save_model(MODEL_NAME,format="cbm",export_parameters=None,pool=None)
