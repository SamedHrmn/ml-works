import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('train.csv')
X_train = data.drop(labels=['Name','Ticket','Fare','Cabin','Embarked','Survived','PassengerId'],axis=1)

test_data = pd.read_csv('test.csv')
X_test = test_data.drop(labels=['Name','Ticket','Fare','Cabin','Embarked','PassengerId'],axis=1)
result_frame = pd.read_csv('gender_submission.csv')


Y_train = data.iloc[:,1:2]



le = LabelEncoder()
X_train['Sex']= le.fit_transform(X_train['Sex'])
X_test['Sex'] = le.fit_transform(X_test['Sex'])

X_train=X_train.apply(lambda x: x.fillna(x.mean()),axis=0)
X_test=X_test.apply(lambda x: x.fillna(x.mean()),axis=0)

"""
ohe = OneHotEncoder()
X_train['Sex'] = pd.DataFrame(ohe.fit_transform(X_train[['Sex']]).toarray())
X_test['Sex'] = pd.DataFrame(ohe.fit_transform(X_test[['Sex']]).toarray())
"""

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)




classifier = Sequential()
classifier.add(Dense(12,kernel_initializer="uniform",activation='relu',input_dim=5))
classifier.add(Dense(12,kernel_initializer="uniform",activation='relu'))
classifier.add(Dense(1,kernel_initializer="uniform",activation='sigmoid'))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
classifier.fit(X_train,Y_train,epochs=100)

y_pred = classifier.predict(X_test_scaled).tolist()

series = []
for val in result_frame['Survived']:
    if val >= 0.5:
        series.append(1)
    else:
        series.append(0)

result_frame['Survived'] = series
result_frame.to_csv('samedharman-submissions.csv', index=False)



