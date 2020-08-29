import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('veriler.csv')

country = data.loc[:,['ulke']].values
labelEncoder = preprocessing.LabelEncoder()
country[:,0] = labelEncoder.fit_transform(data.iloc[:,0])
oneHotEncoder = preprocessing.OneHotEncoder()
country = oneHotEncoder.fit_transform(country).toarray()

gender = data.loc[:,['cinsiyet']].values
labelEncoder = preprocessing.LabelEncoder()
gender[:,0] = labelEncoder.fit_transform(data.iloc[:,-1])
oneHotEncoder = preprocessing.OneHotEncoder()
gender = oneHotEncoder.fit_transform(gender).toarray()



countryFrame = pd.DataFrame(data=country,index=range(len(data)),columns=['fr','tr','us'])
#dummy trap olmaması için gender kolonlarından yalnızca birini aldık.
genderFrame = pd.DataFrame(data=gender[:,1],index=range(len(data)),columns=['cinsiyet'])

data = data.drop(columns=['ulke','cinsiyet'])

resultFrame = pd.concat([countryFrame,data,genderFrame],axis=1)

x1_gender_frame = resultFrame[resultFrame.columns.difference(['cinsiyet'])]
y1_gender_frame = genderFrame

x_gender_train,x_gender_test,y_gender_train,y_gender_test = train_test_split(x1_gender_frame,y1_gender_frame,test_size=0.33,random_state=0)

regressor = LinearRegression()
regressor.fit(x_gender_train,y_gender_train)
y_gender_predict = regressor.predict(x_gender_test)


#boy tahmini
height_x1_frame = resultFrame.loc[:,~resultFrame.columns.isin(['boy'])]
height_y1_frame = pd.DataFrame(data=resultFrame.loc[:,'boy'],index=range(len(data)),columns=['boy'])

x_height_train,x_height_test,y_height_train,y_height_test = train_test_split(height_x1_frame,height_y1_frame,test_size=0.33,random_state=0)
regressor.fit(x_height_train,y_height_train)
y_height_predict = regressor.predict(x_height_test)

#backward elimination
import statsmodels.api as sm

#y = beta + beta1*x1 +beta2*x2 formülündeki beta sabiti için 1 lik bir matris veri kümesine append edilir.
X = np.append(arr=np.ones((22,1)).astype(int),values=x1_gender_frame,axis=1)
result_opt = X[:,:]

# değerler OLS(Ordinary Least Squares) yani en küçük kareler yöntemi ile hesaplanır.
regressor_ols = sm.OLS(exog=result_opt,endog=y1_gender_frame).fit()
print(regressor_ols.summary())

# significant level = sl değeri (ya da p değeri 0,05 olarak alınablir), değerine en yakın değere sahip
# parametreler tek tek çıkartılır.
result_opt = X[:,[0,1,2,3,4,6]]
regressor_ols = sm.OLS(exog=result_opt,endog=y1_gender_frame).fit()
print(regressor_ols.summary())

result_opt = X[:,[0,1,2,3,4]]
regressor_ols = sm.OLS(exog=result_opt,endog=y1_gender_frame).fit()
print(regressor_ols.summary())

result_opt = X[:,[1,2,3,4]]
regressor_ols = sm.OLS(exog=result_opt,endog=y1_gender_frame).fit()
print(regressor_ols.summary())

"""
p_values = regressor_ols.pvalues

high = p_values[0]
sl = 0.05
index = 0


for i in range(1,len(p_values)-1):
    if p_values[i] > high:
        high = p_values[i]
        index = i
        if(high > sl):
            result_opt = np.delete(result_opt, i, 1)
            regressor_ols = sm.OLS(exog=result_opt, endog=y1_gender_frame).fit()
            p_values = regressor_ols.pvalues

print(regressor_ols.summary())

"""
"""
print("PVALUES:"+str(p_values))
sl = 0.005
high_sl = p_values[0]
high_pos = 0

for j in range(1,len(p_values)):
    print("Dis dongu")
    high_sl = p_values[0]
    high_pos = 0
    for i in range(j,len(p_values)-1):
        print("Ic dongu")
        if p_values[i] > high_sl:
            print("p_values[i]: "+str(p_values[i]))
            high_sl = p_values[i]
            high_pos = i
            if high_sl > sl:
                print("Delete calisti.: "+str(high_sl))
                result_opt = np.delete(result_opt,high_pos,1)
                regressor_ols = sm.OLS(exog=result_opt,endog=y1_gender_frame).fit()
                p_values = regressor_ols.pvalues

print(regressor_ols.summary())
"""








