import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from catboost import CatBoostClassifier,Pool
import os
import time
from sklearn.model_selection import train_test_split

# Define word list that use in CountVectorizer
def wordList():
    sonoffWordList = ["sonoff", "ewelink", "ewelinkdemo", "s20", "s26", "b1", "coolkit"]
    tplinkWordList = ["hs100", "lb110", "kc120", "kasa", "tplink", "tp", "link", "tplinkra"]
    hueWordList = ["hue", "philips", "meethue"]
    gurobotWordList = ["gurobot", "aidoor", "wechat"]
    anitechWordList = ["anitech", "nexpie"]
    nestWordList = ["google", "nest", "dropcam"]
    wordList = sonoffWordList + tplinkWordList + hueWordList + gurobotWordList + anitechWordList + nestWordList
    return wordList

def removeChar(detail):
    unwantedChar = [".", ":", "-", "\n", "\t", "(", ")", "="]
    for i in range(0,len(unwantedChar)):
        try:
            for x in range(0,len(detail)):
                if "." or "-" == unwantedChar[i]:
                    detail[x] = detail[x].replace(unwantedChar[i]," ")
                else:
                    detail[x] = detail[x].replace(unwantedChar[i],"")
                print(x)
        except:
            pass
            print("*****************************************")
    return detail

TRAIN_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_dataset_clean.csv")
TEST_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dataset_clean.csv")
# Train dataset
headernames = ['ipSrc', 'ipDst', 'macSrc', 'macDst', 'portSrc', 'portDst', 'pktLength', 'deviceName', 'protocol','flagmDNS', 'detail' ]
dataset = pd.read_csv(TRAIN_DATASET, names = headernames)
dataset.head()

X_train = dataset.drop(['deviceName'],axis=1)
X_train = X_train.drop(['detail'],axis=1)
y_train = dataset['deviceName']

#########################################################################

testDataset = pd.read_csv(TEST_DATASET, names = headernames)
testDataset.head()

X_test = testDataset.drop(['deviceName'],axis=1)
X_test = X_test.drop(['detail'],axis=1)
y_test = testDataset['deviceName']

try :
    X_train.to_csv(X_train, index = False, header = False)
    X_test.to_csv(X_test, index = False, header = False)
except:
    pass

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=4)
categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
# IMPORTANT! : cannot use two CatBoost pool.
#eval_pool = Pool(X_test, y_test)
pool = Pool(X_train, y_train,cat_features=categorical_features_indices)

params = {
'iterations': 9,
'learning_rate': 0.4,
'eval_metric': 'Accuracy',
'l2_leaf_reg':2.8,
'use_best_model': False
}
classifier = CatBoostClassifier(**params)
start = time.time()
#classifier.fit(pool , eval_set = (X_test,y_test),early_stopping_rounds=10)
classifier.fit(pool)

stop = time.time()

start2 = time.time()
y_pred = classifier.predict(X_test)
stop2 = time.time()


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
cm = cm.astype(np.float64) / cm.sum(axis=1)[:, np.newaxis]
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
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
