import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('Position_Salaries.csv')

X = data.iloc[:,1:2]
Y = data.iloc[:,-1:]

# estimators = kaç decision tree çizileceği bilgisi
regressor = RandomForestRegressor(random_state=0,n_estimators=10)
regressor.fit(X,Y.values.ravel())

plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.show()