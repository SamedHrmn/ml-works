import numpy

np_array = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# ndim ve shape
print("Dizi boyutu: " + str(np_array.ndim))
print("\nSatir sutun sayisi: " + str(np_array.shape))

np_array = np_array.reshape(5, 2)
print("\nYeniden boyutlandirildi:\n " + str(np_array))
print("\nSatir sutun sayisi: " + str(np_array.shape))

# arange() -> başlangıç ve bitiş değerleri arasında belirtilen
# artış değeri ile bir numpy dizisi döndürür.
print("\n" + str(numpy.arange(0, 15, 3)))

# satır ve sütun seçme işlemleri
first_seconds_row = np_array[0:2]
first_seconds_column = np_array[:, 0:2]
first_column = np_array[:, 0]
print("\nIlk iki satir : " + str(first_seconds_row))
print("\nIlk iki sutun : " + str(first_seconds_column))
print("\nIlk sutun : " + str(first_column))

print("\nDiziyi ters cevirme : " + str(np_array[::-1]))
print("\nBirim matris olusturmak : " + str(numpy.eye(3, 3)))

arr1 = numpy.array([[2, 5], [5, 8], [9, 11]])
arr2 = numpy.array([[5, 1], [-3, 8], [-9, -11]])
print("\nDiziyi tek boyuta indirme : " + str(arr1.ravel()))

print("\nSatir bazli matris birlestirme : " + str(numpy.concatenate([np_array, np_array], axis=0)))
print("\nSutun bazli matris birlestirme : " + str(numpy.concatenate([np_array, np_array], axis=1)))

print("\nYatay matris birlestirme : " + str(numpy.hstack((arr1, arr2))))
print("\nDikey matris birlestirme : " + str(numpy.vstack((arr1, arr2))))

# %% matematiksel islemler
import numpy

a = numpy.array([[0, 1, 2, 3], [3, 5, 7, 4]])
b = numpy.array([[5, 6, 7, 8], [2, 5, 8, 3]])

# sum() , min() , max()
print("Toplam : " + str(a.sum()))
print("\nMin : " + str(a.min()))
print("\nMax : " + str(a.max()))
print("\nSutun toplam : " + str(a.sum(axis=0)))
print("\nSatir toplam : " + str(a.sum(axis=1)))

# median() , var() , std()
print("\nMedyan : " + str(numpy.median(a)))
print("\nVaryans : " + str(a.var()))
print("\nStandart sapma : " + str(a.std()))

# dot() , T
print("\nMatris Carpim : " + str(numpy.dot(a, b.T)))

print("\nMatris ici kosul : " + str(b > 5))

print("\nSin : " + str(numpy.sin(b)))
print("\nCos : " + str(numpy.cos(b)))
print("\nLog : " + str(numpy.log(b)))
print("\nKok : " + str(numpy.sqrt(b)))
print("\nExponansiyeli : " + str(numpy.exp(b)))
