import pandas
import numpy

numbers = [0,1,2,3,4,5]
numpy_array = numpy.array(numbers)
print(numpy_array)

# pandas serilerinde indexleme yapısı.
pandas_index = ['a','b','c','d','e','f']
pandas_series = pandas.Series(data=numbers,index=pandas_index)
print(pandas_series)

dictionary_index = {'sifir':0,'bir':1,'iki':2,'uc':3,'dort':4,'bes':5}
pandas_series = pandas.Series(data=dictionary_index)
print(pandas_series)

# median()
print("\nMedyan degeri: " + str(pandas_series.median()))

# reshape()
print("\nSatir sutun degistirmek:\n" + str(pandas_series.values.reshape(2,3)))

# append()
pandas_series = pandas.Series(data=pandas_series.append(pandas_series))
print("\nMatris birlestirme:\n"+ str(pandas_series.values.reshape(2,6)))

# DataFrame veri yapısı.

s1 = pandas.Series(['Samed','Orhan','Ferdi'])
s2 = pandas.Series(['Harman','Gencebay','Tayfur'])

data = dict(Ad=s1,Soyad=s2)
dFrame = pandas.DataFrame(data)
print("\nDataFrame:\n" + str(dFrame))

covid_dframe = pandas.read_csv('covid_19_data.csv')
death_filter_result = covid_dframe.sort_values(by='Deaths',ascending=False).head(10)
turkey_frame = covid_dframe[covid_dframe['Country/Region']=='Turkey'].sort_values(by='Recovered',ascending=False)