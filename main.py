from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

api = KaggleApi()
api.authenticate()
api.dataset_download_file('uciml/breast-cancer-wisconsin-data', file_name = 'data.csv')

#Reading in our data set
df = pd.read_csv('data.csv')

#Lets take a look at the dimensions of our data set
print(df.shape)
print()
#It looks like we have 33 columns, and 569 rows

#Lets take a look at what our columns represent
print(df.columns)
print()

#we will not be needing the 'id' or 'Unnamed: 32' columns so lets drop those now
df = df.drop(['id'], axis = 1)
print(df.columns)
print()
df = df.drop(['Unnamed: 32'], axis = 1)
print(df.columns)
print()

#We want to use features to predict a diagnosis, so lets look at the possible diagnosis results
print(df['diagnosis'].value_counts())
#We see our two diagnoses are B (benign) and M(malignant), coming in at 63 and 37 percent, respectively. Since we are
#attempting to predict into one of two classes, we will use logistic regression

#Now lets convert our label (diagnosis) and our features (everything else) into arrays
X = np.array(df.drop(['diagnosis'], axis = 1))
y = np.array(df['diagnosis'])

print(X[:2])
print()
print(y[:2])
print()

#Now lets split our data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .176)

#Now we scale our features, since they take up a very broad range of numbers
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Now lets train our model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Comparing our y_pred to our y_test
y_pred = classifier.predict((X_test))
comparison = []
for i in range(len(y_test)):
    a_list = []
    a_list.append(y_pred[i])
    a_list.append(y_test[i])
    comparison.append(a_list)
print('Comparison (prediction, actual):', comparison)
print()
print(classifier.score(X_test, y_test))

#We have achieved a consistent accuracy of around 98%, which shows that our model is a good predecitor of benign or
#malignant cancer cells, given the available features. 