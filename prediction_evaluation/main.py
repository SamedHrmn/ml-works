import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

data = pd.read_csv('iris.csv')
data = data.drop(['Id','SepalWidthCm','Species'],axis=1)

corr = data.corr()

X = data.iloc[:,0:1]
Y = data.iloc[:,-1:]

#--------Linear Regression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)
lin_model = sm.OLS(exog=lin_reg.predict(X),endog=Y)
print("Linear Model : \n")
print(lin_model.fit().summary())

#--------Polynomail Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,Y)
poly_model = sm.OLS(exog=lin_reg.predict(poly_reg.fit_transform(X)),endog=Y)
print("Poly Model : \n")
print(poly_model.fit().summary())


#---------Support Vector Regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
sc = StandardScaler()
x_scaled = sc.fit_transform(X)
y_scaled = np.ravel(sc.fit_transform(Y.values.reshape(-1,1)))

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scaled,y_scaled)
svr_model = sm.OLS(exog=svr_reg.predict(x_scaled),endog=y_scaled)
print("SVR Model : \n")
print(svr_model.fit().summary())

import matplotlib.pyplot as plt
plt.scatter(x_scaled,y_scaled)
plt.plot(x_scaled,svr_reg.predict(x_scaled))
plt.show()


#---------Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X,Y)
dt_model = sm.OLS(exog=dt_reg.predict(X),endog=Y)
print("Decision Tree Model : \n")
print(dt_model.fit().summary())


#---------Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,np.ravel(Y))
rf_model = sm.OLS(exog=rf_reg.predict(X),endog=Y)
print("Random Forest Model : \n")
print(rf_model.fit().summary())

