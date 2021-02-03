import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:,1:2]
Y= data.iloc[:,-1:]

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

plt.scatter(X,Y,color ="red")
plt.plot(X,regressor.predict(X),color = "blue")
plt.show()

