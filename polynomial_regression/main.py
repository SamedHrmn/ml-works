import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("Position_Salaries.csv")
x = data.iloc[:,1:2]
y = data.iloc[:,-1:]
X = x.values
Y= y.values

# lineer regresyon ve görselleştirme
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
plt.scatter(X,Y)
plt.plot(x,lin_reg.predict(X))
plt.show()

#2. derece polinom
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
# polinomun fit edilip görselleştirilmesi.
lin_reg.fit(x_poly,Y)
plt.scatter(X,Y)
plt.plot(x,lin_reg.predict(x_poly))
plt.show()
