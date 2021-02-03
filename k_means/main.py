from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('musteriler.csv')
X = data.iloc[:,3:]


results = []
for i in range(1,10): # Doğru cluster sayısını görmek için deneme
    kmeans = KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(X)
    results.append(kmeans.inertia_)

plt.plot(range(1,10),results) # sonuçları çizdir , dirsek noktası uygun nokta olur.
plt.show()
