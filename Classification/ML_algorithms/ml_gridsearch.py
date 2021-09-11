import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from catboost import CatBoostClassifier,Pool
import os
import time
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV

TRAIN_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model3_dataset", "model3_train_paper.csv")
TEST_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model3_dataset", "model3_test_paper.csv")
# Train dataset
headernames = ['ipAddr', 'mac', 'portSrc', 'portDst', 'pktLength', 'deviceName', 'protocol', 'detail' ]
dataset = pd.read_csv(TRAIN_DATASET, names = headernames)
dataset.head()

#########################################################################

# Seperate Dataset

testDataset = pd.read_csv(TEST_DATASET, names = headernames)
testDataset.head()
X_train = dataset.drop(['deviceName'],axis=1)
X_train = X_train.drop(['detail'],axis=1)

y_train = dataset['deviceName']
X_test = testDataset.drop(['deviceName'],axis=1)
X_test = X_test.drop(['detail'],axis=1)
y_test = testDataset['deviceName']

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
# IMPORTANT! : cannot use two CatBoost pool.
print X_train
pool = Pool(X_train, y_train,cat_features=categorical_features_indices)

array = []
i = 50
while i < 500:
    array.append(i)
    i += 50

depthArray = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

classifier = CatBoostClassifier()
params = {'iterations': [300,100],
          'learning_rate': [0.46],
          'logging_level': ['Verbose'],
          'l2_leaf_reg': [2]
         }

best_param = classifier.grid_search(params,pool)
print(best_param)
classifier = CatBoostClassifier(**best_param)
classifier.fit(pool)

start2 = time.time()
y_pred = classifier.predict(X_test)
stop2 = time.time()


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
#cm = cm.astype(np.float64) / cm.sum(axis=1)[:, np.newaxis]
#np.set_printoptions(formatter={'float_kind':'{:f}'.format})
print("Confusion Matrix:")
print(cm)
report = classification_report(y_test, y_pred)
#mapping = dict(zip(labelencoder.classes_, range(1, len(labelencoder.classes_)+1)))
#print(mapping)
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

#classifier.save_model(model_name,format="cbm",export_parameters=None,pool=None)
