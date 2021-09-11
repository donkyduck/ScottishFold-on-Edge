import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from catboost import CatBoostClassifier,Pool
import os
#import category_encoders as ce
import time

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
        except:
            pass
    return detail

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_encoded.csv")
bow = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bow.csv")
text = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_xtratrees.txt")
model_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.cbm")
# Train dataset
headernames = ['ipSrc', 'ipDst', 'macSrc', 'macDst', 'portSrc', 'portDst', 'pktLength', 'deviceName', 'protocol', 'detail' ]
# for ipv4.csv
#headernames = ['ipSrc', 'ipDst', 'macSrc', 'macDst', 'portSrc', 'portDst', 'pktLength', 'protocol', 'detail', 'deviceName' ]
#listOfStrings = wordList()
dataset = pd.read_csv(path, names = headernames)

#bagofwordsDataframe = pd.read_csv(bow, names = listOfStrings)
dataset.head()
#bagofwordsDataframe.head()
#print(dataset.loc[0])
#deviceName = pd.get_dummies(dataset['deviceName'],prefix='deviceName')

X = dataset.drop(['deviceName'],axis=1)
X = X.drop(['detail'],axis=1)
"""
if len(bagofwordsDataframe) != len(detail):
    print(len(detail))
    detail = removeChar(detail)
    vectorizer = CountVectorizer(vocabulary=listOfStrings)
    bagofwords = vectorizer.fit_transform(detail).toarray()
    bagofwordsDataframe = pd.DataFrame(data = bagofwords, columns = listOfStrings)
    bagofwordsDataframe.to_csv(bow, index = False, header = False)
    print(len(bagofwordsDataframe))
"""
"""
example >> print : anitech nexpie dns query
print(detail[14569])
print(bagofwordsDataframe.loc[[14569]])
"""
#ipSrcDummies = pd.get_dummies(dataset['ipSrc'],prefix='ipSrc')
#ipDstDummies = pd.get_dummies(dataset['ipDst'],prefix='ipDst')
#protocol = pd.get_dummies(dataset['protocol'],prefix='protocol')
#print type(protocol)
#macSrcDummies = pd.get_dummies(dataset['macSrc'],prefix='macSrc')
#macDstDummies = pd.get_dummies(dataset['macDst'],prefix='macDst')
#portSrcDummies = pd.get_dummies(dataset['portSrc'],prefix='portSrc')
#portDstDummies = pd.get_dummies(dataset['portDst'],prefix='portDst')

#labelencoder = LabelEncoder()
#print(deviceName.columns.tolist())
#print type(dataset['ipSrc'])
#print type(ipSrcDummies)
"""
Dict : DeviceName
{0: "Anitech", 1: "B1", 2: "GUROBOT", 3: "HS110", 4: "HUE", 5: "KC120", 6: "LB100", 7: "NESTCAM", 8: "S20", 9: "S26"}
"""

#ipSrcDummies,ipDstDummies,portSrcDummies,portDstDummies,bagofwordsDataframe,macSrcDummies,macDstDummies,pktLength,protocol
#X = newpktLength
#X = X.join([ipDst])

#y = dataset['deviceNumber'] = labelencoder.fit_transform(dataset['deviceName'])
#print(list(X))
y = dataset['deviceName']


"""
OUT OF MEMORY, THEN WE NEED MONEY FOR UPGRADE MEMORY :)
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=4)
"""
NOTE : n_estimators = x, random_state=x) | test_size = 0.25, random_state=4
- n_estimators = 1 | Acc. = 0.9779570743025893 | TrainingTime = 4.5 sec. | random_state=5 Acc. = 0.9808047249907715
- n_estimators = 36 | Acc. = 0.9867900648631546 | TrainingTime = 59 sec - 1 min. | random_state=7 Acc. = 0.9871064704951749
- n_estimators = 75 | Acc. = 0.9871064704951749 | traningTime - 2 min. | Acc. = 0.9869482676791647
- n_estimators = 100 | random_state=9 Acc. = 0.9872383061751833 | TrainingTime = 160.1 sec. | Bootstrap=False = 0.9871592047671782, 2x0.9877920160312187,
0.987686547487212,2x0.9877392817592153,0.9873174075831883
** CatBoost  0.9803828508147445 /900|0.900015 def 0.5
** CatBoost  0.9808574592627749 /1000|0.900015
** CatBoost  0.981358434846807 /1100|0.900015
** CatBoost 0.9850234667510415 /2000|0.900015
** CatBoost 0.9857090122870854 /3000|0.900015
******
100/depth=6/learning_rate=0.900015 : 0.9617149185255498
, early_stopping_rounds=10
iterations=100,depth=6 | learning_rate = x
0.9629805410536307 0.91
0.9616885513895481 0.88
0.9606074988134788 0.83
0.9600537889574434 0.84
0.75 overfit detected
**********************
0.9839951484469757 | iterations=1485,learning_rate=0.91
0.9862890892791225 | iterations=2954,learning_rate=0.88
0.9862890892791225 | iterations=3852,learning_rate=0.72,l2_leaf_reg=3
0.9856035437430787 | iterations=4279,learning_rate=0.5
*****
0.9579444180773085 | 2.9
0.9589727363813743 | 2.8
0.9582344565733270 | 2.79
0.9579971523493118 | 2.78
0.958735432157359 | 2.83
"""
categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
# IMPORTANT! : cannot use two CatBoost pool.
#eval_pool = Pool(X_test, y_test)
pool = Pool(X_train, y_train,cat_features=categorical_features_indices)

n_estimators_no = 100
#classifier = CatBoostClassifier(iterations=10,learning_rate=0.5,l2_leaf_reg=2.8)
params = {
'iterations': 100,
'learning_rate': 0.5,
'eval_metric': 'Accuracy',
'l2_leaf_reg':2.8,
'use_best_model': False
}
classifier = CatBoostClassifier(**params)
#classifier = RandomForestClassifier(n_estimators = n_estimators_no,random_state=9)
start = time.time()
classifier.fit(pool , eval_set = (X_test,y_test),early_stopping_rounds=10) # Default (both random forest and meowboost)
#classifier.fit(X, y)
#classifier.fit(X_train, y_train, eval_set = eval_pool, early_stopping_rounds=10) # w/ overfit detector
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

classifier.save_model(model_name,format="cbm",export_parameters=None,pool=None)

df_cm = pd.DataFrame(cm, index = [i for i in "ABCDEFGHIJ"], columns = [i for i in "ABCDEFGHIJ"])
plt.figure(figsize = (40,40))
#sn.heatmap(df_cm, annot=True,fmt='2g',linewidth=.5)
#plt.show()
