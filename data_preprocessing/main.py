import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#veri ön işleme aşaması, verilerin makine öğrenmesi
#algoritmalarına hazırlandığı aşamadır.

#verinin okunması aşaması.
data = pd.read_csv('veriler.csv')

#verideki eksik kısımların tamamlanması, loc metodu ile.
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
age = data.loc[:,['yas']]
imputer = imputer.fit(age)
age = imputer.transform(age)
print(age)


#iloc metodu ile çekersek
'''
age = data.iloc[:,1:4].values
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])
print(age)
'''

#kategorik verilerin nümerik verilere dönüşümü.

country = data.loc[:,['ulke']].values
labelEncoder = preprocessing.LabelEncoder()
country[:,0] = labelEncoder.fit_transform(data.iloc[:,0])
oneHotEncoder = preprocessing.OneHotEncoder()
country = oneHotEncoder.fit_transform(country).toarray()

#numpy dizilerinin dataframe'e dönüştürülmesi.
countryFrame = pd.DataFrame(data=country, index= range(len(data)),columns=['fr','tr','us'])
ageFrame = pd.DataFrame(data=age,index=range(len(data)),columns=['yas'])

data[['yas']] = ageFrame['yas'].values
data = data.drop(columns="ulke")

resultFrame = pd.concat([countryFrame,data],axis=1)
print(resultFrame)

#
x1_frame = resultFrame
y1_frame = resultFrame.iloc[:,-1]
x1_frame = x1_frame.drop(columns='cinsiyet')
print(x1_frame)
print(y1_frame)

#verilerin eğitim ve test olarak ayrılması.
x_train,x_test,y_train,y_test = \
    train_test_split(x1_frame,y1_frame,test_size=0.33,random_state=0)

#veri ölçeklemesi.
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


