import numpy as np
"""
This utility functions are for preparing data of IoT device detection (from Arunan) for train and test
"""
# Train the IoT data with Catboost
# Load data from a csv file to DataFrame

def ListDevice(filename):
    """
    This function is used to create the list of IoT devices mapping with their MAC addresses
    :param filename:
    :return: dataframe of list IoT devices
    """
    import pandas as pd
    with open(filename) as x :
        data = [line.split('\t') for line in x]

    data2 = []
    for r in data:
        row2 = []
        for x in r :
            if x != "":
                row2.append(x)

        data2.append(row2)
    df = pd.DataFrame(data2)
    df2 = df.drop(columns=2)
    df2.columns = ["device_name","MAC_address"]
    return df2

def LoadIoTData(file_dir):
    """
    This program is to load csv files in file_dir to dataframe
    :param file_dir: file directory
    :return: IoT_df1 : dataframe of IoT features
    """""
    import pandas as pd
    import os

    IoT_df1 = pd.DataFrame(columns=["Packet ID","Time","Size","eth.src","eth.dst","IP.src","IP.dst","IP.proto","port.src", "port.dst"])
    for r,d,f in os.walk(file_dir):
        for file in f:
            if file.endswith("16-09-23.csv"):
                entry = os.path.join(r,file)
                IoT_df2 = pd.read_csv(entry)
                IoT_df1 = pd.concat([IoT_df1,IoT_df2],ignore_index=True)
    return IoT_df1

def findexact(lst,key):
    """
    This function is to find which "index" in "lst" matched the data in "key"
    :param lst: list of array
    :param key: data that is wanted to matched
    :return: idx : index of "lst" where its value matched the data in "key"
    """
    for idx , elem in enumerate(lst):
        if key == elem:
            return idx
def addY_label(IoT_df1,df2):
    """
    This function is to add Y_label from df2 to IoT_df1
    :param IoT_df1: dataset x_val
    :param df2: dataset for y_val
    :return: y_val
    """
    y_label = []
    macADD_list = df2["MAC_address"].tolist()
    macADD_list = [x.strip(' ') for x in macADD_list]

    for r in range(len(IoT_df1)):
        src_device = findexact(macADD_list,IoT_df1.iloc[r]['eth.src'])
        dst_device = findexact(macADD_list, IoT_df1.iloc[r]['eth.dst'])
        if src_device != 30:
            y_label.append(src_device)
        elif dst_device != 30:
            y_label.append(dst_device)
        else:
            print("null")
    return y_label

def rd_X_data(df_iot,Ratio,Ran_state):
    """
    This function is to random dataset for x_train
    :param df_iot: dataset
    :param Ratio: percentage of data in used
    :param Ran_state: random state
    :return: x_train
    """
    data_train = df_iot.sample(frac=Ratio,random_state=Ran_state)
    return data_train
def prepare_dataforTrain_Test(data_size,List_device_file_path,file_dir,train_ratio,Rand_state):
    List_dev_df = ListDevice(List_device_file_path)
    IoT_df = LoadIoTData(file_dir)
    y_label = addY_label(IoT_df[0:data_size], List_dev_df)
    x_df = IoT_df[0:data_size]
    x_df.insert(0, "Device_id", y_label, True)
    x_train = rd_X_data(x_df, train_ratio, Rand_state)
    x_test = rd_X_data(x_df, 1-train_ratio, Rand_state)
    y_train = x_train["Device_id"]
    y_test = x_test["Device_id"]
    x_train_new = x_train.drop(["Device_id"], axis=1)
    x_test_new = x_test.drop(["Device_id"], axis=1)
    """
    print x_train_new['IP.src']
    print("***********************************************************")
    print y_train
    print("***********************************************************")
    print x_test_new
    print("***********************************************************")
    print y_test
    exit()
    """
    return x_train_new, y_train,  x_test_new, y_test

def CleanData_CB(X,drop_feature):
        """
        This function is to clean the data for training and testing with CatBoost.
        X : dataframe type for train
        y : dataframe type for test
        drop_feature : feature name that is wanted to be dropped , e.g., ['Time']
        """
        import numpy as np
        import pandas as pd


        x_new = X.drop(drop_feature,axis=1)

        return x_new


def CB_train(params,x_train,y_train):
    """
    This function is for CatBoost Training
    """
    import numpy as np
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    from catboost import CatBoostClassifier,Pool
    import category_encoders as ce
    import pandas as pd
    import time
    import os

    from sklearn.model_selection import train_test_split
    path = "/home/meowmeow/Desktop/github/ScottishFold/ml/label_encoded.csv"
    headernames = ['ipSrc', 'ipDst', 'macSrc', 'macDst', 'portSrc', 'portDst', 'pktLength', 'deviceName', 'protocol', 'detail' ]
    # for ipv4.csv
    #headernames = ['ipSrc', 'ipDst', 'macSrc', 'macDst', 'portSrc', 'portDst', 'pktLength', 'protocol', 'detail', 'deviceName' ]
    dataset = pd.read_csv(path, names = headernames)
    dataset.head()
    #print(dataset.loc[0])
    #deviceName = pd.get_dummies(dataset['deviceName'],prefix='deviceName')

    X = dataset.drop(['deviceName'],axis=1)
    X = X.drop(['detail'],axis=1)
    y = dataset['deviceName']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
    eval_pool = Pool(X_train, y_train,cat_features=categorical_features_indices)
    print X_train
    n_estimators_no = 100
    classifier = CatBoostClassifier(**params)
    start = time.time()
    classifier.fit(eval_pool)
    stop = time.time()
    return eval_pool, classifier, start, stop

def Plot_Confusion(Fig_size,font_size,label_size,fig_name):

    from sklearn.metrics import plot_confusion_matrix
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(Fig_size[0], Fig_size[1]))
    ax.set_title('Confusion metrix',fontweight="bold", size=font_size) # Title
    ax.tick_params(axis='both', which='major', labelsize=label_size)  # Adjust to fit
    ax.xaxis.set_ticklabels(['False', 'True']);
    ax.yaxis.set_ticklabels(['False', 'True']);
    plt.rcParams.update({'font.size': font_size})
    plot_confusion_matrix(classifier, x_test2, y_test, normalize='true', cmap=plt.cm.Blues, ax=ax)
    plt.show()
    fig.savefig(fig_name)

def CB_train_feature_rank(train_pool, x_train, model):
    """This function is to check which feature is the most importance by ranking the most importance"""

    import numpy as np
    from catboost import CatBoostClassifier,Pool
    import category_encoders as ce
    import pandas as pd

    categorical_features_indices = np.where(x_train.dtypes != np.float)[0]
    train_pool = Pool(x_train2, y_train, cat_features=categorical_features_indices)
    feature_importances = model.get_feature_importance(train_pool)
    feature_names = x_train2.columns
    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
        print('{}: {}'.format(name, score))

    return name, score


## Pre-process loading Dataset to x_train, y_train, x_test, y_test
List_device_file_path = "/home/meowmeow/Downloads/List_Of_Devices.txt"
file_dir = "/home/meowmeow/Downloads/"

Rand_state = 1 # Random state for sampling the data from the dataframe.
train_ratio = 0.7 # split ration
data_size = 8000  # select the number of data packets
List_device_file_path #
x_train_new, y_train,x_test_new, y_test = prepare_dataforTrain_Test(data_size,List_device_file_path,file_dir,train_ratio,Rand_state)


drop_feature = ['Time']
x_train2 = CleanData_CB(x_train_new,drop_feature)
x_test2 = CleanData_CB(x_test_new,drop_feature)
params = {
    'iterations': 10,
    'learning_rate': 0.1,
    'eval_metric': 'Accuracy',
    'random_seed': 42,
    'use_best_model': False
}
train_pool, classifier, start, stop = CB_train(params, x_train2,y_train)

print(y_train)
import time
start2 = time.time()
y_pred = classifier.predict(x_test2)
stop2 = time.time()
model_name = 'IoT_device_catboost_model_01.cbm'
classifier.save_model(model_name,format="cbm",export_parameters=None,pool=None)

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

fig_name = '/home/meowmeow/Downloads/cm.png'
Fig_size = [40,40]
font_size = label_size = 24
#Plot_Confusion(Fig_size,font_size,label_size,fig_name)

name, score = CB_train_feature_rank(train_pool, x_train2, classifier)
