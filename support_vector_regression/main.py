import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("Position_Salaries.csv")
x = data.iloc[:,1:2]
y = data.iloc[:,-1:]

X = x.values
Y = y.values

sc_X = StandardScaler()
x_scaled = sc_X.fit_transform(X)

sc_Y = StandardScaler()
y_scaled = sc_Y.fit_transform(Y.reshape(-1,1))
y_scaled = np.ravel(y_scaled)


regressor = SVR(kernel="rbf")
regressor.fit(x_scaled,y_scaled)

plt.scatter(x_scaled,y_scaled,color="red")
plt.plot(x_scaled,regressor.predict(x_scaled),color="blue")
plt.show()