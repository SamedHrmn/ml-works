import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = pd.read_csv('satislar.csv')
months = data[['Aylar']]
sales = data[['Satislar']]

x_test,x_train,y_test,y_train =\
    train_test_split(months,sales,train_size=0.33,random_state=0)


"""
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""

lr = LinearRegression()
lr.fit(x_train,y_train)

prediction = lr.predict(x_test)


plt.figure(1)
plt.scatter(x_train,y_train,c="blue")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.title("Aylar-Satışlar Grafiği")
plt.plot(x_train,lr.predict(x_train),'r')
plt.show()


print("r2: "+str(r2_score(y_test,prediction)))

