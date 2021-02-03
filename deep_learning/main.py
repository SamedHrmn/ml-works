import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

data = pd.read_csv('Churn_Modelling.csv')

X = data.iloc[:,3:13].values
Y = data.iloc[:,-1:].values

le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
X[:,2] = le.fit_transform(X[:,2])

ohe = ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],remainder="passthrough")
X = ohe.fit_transform(X)
X = X[:,1:]

x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

classifier = Sequential()
classifier.add(Dense(6,kernel_initializer="uniform",activation='relu',input_dim=11))
classifier.add(Dense(6,kernel_initializer="uniform",activation='relu'))
classifier.add(Dense(1,kernel_initializer="uniform",activation='sigmoid'))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
classifier.fit(X_train,y_train,epochs=50)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test,y_pred)
print(cm)